import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_dim=64, max_period=10000):
        super().__init__()
        self.time_dim = time_dim
        self.max_period = max_period

    def forward(self, t):
        device = t.device
        half_dim = self.time_dim // 2
        freqs = -math.log(self.max_period) * torch.arange(half_dim, device=device).float() / half_dim
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
        return emb


class AtomStateEmbedding(nn.Module):
    """Embed discrete atom type At and charge + time into scalar node features."""
    def __init__(self, hidden_dim=256, atom_emb_dim=64, time_dim=64, num_atom_types=13):
        super().__init__()
        self.attr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.atom_state_emb = nn.Embedding(num_atom_types, atom_emb_dim)
        self.atom_input_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2 + atom_emb_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, St, At, t_emb):
        B, M, _ = St.shape
        attr = St[:, :, 3:]
        attr_emb = self.attr_proj(attr)
        at_emb = self.atom_state_emb(At)
        t_atom = t_emb[:, None, :].expand(B, M, -1)
        atom_input = torch.cat([attr_emb, at_emb, t_atom], dim=-1)
        x_att = self.atom_input_proj(atom_input)
        return x_att


class RBFDistance(nn.Module):
    def __init__(self, n_rbf=48, d_min=0.0, d_max=12.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.d_min = d_min
        self.d_max = d_max
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        width = (d_max - d_min) / n_rbf
        self.register_buffer("width", torch.tensor(width))

    def forward(self, d):
        diff = d.unsqueeze(-1) - self.centers
        rbf = torch.exp(-((diff / self.width) ** 2))
        return rbf


class BondStateEmbedding(nn.Module):
    def __init__(self, hidden_dim=256, bond_emb_dim=64, time_dim=64,
                 num_bond_types=6, n_rbf=48, d_min=0.0, d_max=12.0):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.bond_state_emb = nn.Embedding(num_bond_types, bond_emb_dim)
        self.rbf = RBFDistance(n_rbf=n_rbf, d_min=d_min, d_max=d_max)
        self.bond_input_proj = nn.Sequential(
            nn.Linear(bond_emb_dim + n_rbf + 1 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, St, Bt, t_emb):
        B, M, _ = St.shape
        bond_emb = self.bond_state_emb(Bt)
        x_pos = St[:, :, :3]
        d_raw = torch.cdist(x_pos, x_pos)
        rbf = self.rbf(d_raw)
        d_feat = (d_raw / self.d_max).clamp(max=1.0).unsqueeze(-1)
        t_edge = t_emb[:, None, None, :].expand(B, M, M, -1)
        edge_input = torch.cat([bond_emb, rbf, d_feat, t_edge], dim=-1)
        e = self.bond_input_proj(edge_input)
        return e


class VEncoder(nn.Module):
    def __init__(self, hidden_dim=256, n_rbf=48, d_min=0.0, d_max=12.0):
        super().__init__()
        # Kept as `k_att_proj` (not `v_proj`) so this attribute's name - and
        # therefore its checkpoint state_dict key - matches already-trained
        # checkpoints. Purely a legacy name; it projects V's (site) attributes.
        self.k_att_proj = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, V, site_valid):
        v_pos = V[:, :, :3]
        v_attr = V[:, :, 3:]
        v_att = self.k_att_proj(v_attr)
        v_att = v_att.masked_fill(~site_valid.unsqueeze(-1), 0.0)
        v_pos = v_pos.masked_fill(~site_valid.unsqueeze(-1), 0.0)
        return v_pos, v_att


class VtoSLayer(nn.Module):
    """
    V -> S conditioning layer.
    """
    def __init__(
        self,
        a_hidden_dim=256,
        k_hidden_dim=256,
        edge_dim=256,
        hidden_dim=256,
        n_rbf=48,
        d_min=0.0,
        d_max=12.0,
        coord_scale=0.1,
    ):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.coord_scale = coord_scale
        self.rbf = RBFDistance(n_rbf=n_rbf, d_min=d_min, d_max=d_max)
        msg_scalar_in = a_hidden_dim + k_hidden_dim + n_rbf + 1
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_scalar_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(a_hidden_dim + hidden_dim, a_hidden_dim), nn.SiLU(),
            nn.Linear(a_hidden_dim, a_hidden_dim),
        )
        self.node_norm = nn.LayerNorm(a_hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + a_hidden_dim * 2 + n_rbf + 1, edge_dim),
            nn.SiLU(), nn.Linear(edge_dim, edge_dim),
        )
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, x_pos, x_att, e, v_pos, v_att, atom_valid, site_valid):
        B, M, D = x_att.shape
        K_n = v_att.shape[1]
        K_D = v_att.shape[-1]
        diff = x_pos[:, :, None, :] - v_pos[:, None, :, :]
        d_raw = torch.norm(diff, dim=-1)
        rbf = self.rbf(d_raw)
        d_feat = (d_raw / self.d_max).clamp(max=1.0).unsqueeze(-1)
        x_i = x_att[:, :, None, :].expand(B, M, K_n, D)
        v_a = v_att[:, None, :, :].expand(B, M, K_n, K_D)
        msg_scalar = torch.cat([x_i, v_a, rbf, d_feat], dim=-1)
        pair_valid = atom_valid[:, :, None] & site_valid[:, None, :]

        m = self.msg_mlp(msg_scalar)
        gate = torch.sigmoid(self.gate_mlp(m))
        m = m * gate
        m = m.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        denom = pair_valid.sum(dim=2, keepdim=True).clamp(min=1).float()
        v_ctx = m.sum(dim=2) / denom

        # NFU
        ns_in = torch.cat([x_att, v_ctx], dim=-1)
        ds = self.node_mlp(ns_in)
        x_att = self.node_norm(x_att + ds)
        x_att = x_att.masked_fill(~atom_valid.unsqueeze(-1), 0.0)

        # NPU (reuses message m from above, before edges are updated,
        # so edges see the new positions)
        coord_w = self.coord_mlp(m)
        coord_w = coord_w.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        unit = diff / (d_raw.unsqueeze(-1) + 1.0)
        dpos = (unit * coord_w).sum(dim=2) / denom
        x_pos = x_pos + self.coord_scale * dpos
        x_pos = x_pos.masked_fill(~atom_valid.unsqueeze(-1), 0.0)

        # EFU
        eye = torch.eye(M, device=x_att.device, dtype=torch.bool).view(1, M, M)
        pair_valid_mm = atom_valid[:, :, None] & atom_valid[:, None, :] & (~eye)
        dmm = torch.cdist(x_pos, x_pos)
        rbf_mm = self.rbf(dmm)
        dmm_feat = (dmm / self.d_max).clamp(max=1.0).unsqueeze(-1)
        si = x_att[:, :, None, :].expand(B, M, M, D)
        sj = x_att[:, None, :, :].expand(B, M, M, D)
        edge_in = torch.cat([e, si, sj, rbf_mm, dmm_feat], dim=-1)
        de = self.edge_mlp(edge_in)
        de = de.masked_fill(~pair_valid_mm.unsqueeze(-1), 0.0)
        e = self.edge_norm(e + de)
        e = e.masked_fill(~pair_valid_mm.unsqueeze(-1), 0.0)
        return x_pos, x_att, e


class StoSLayer(nn.Module):
    """S <-> S self-update layer"""
    def __init__(
        self,
        hidden_dim=256,
        edge_dim=256,
        n_rbf=48,
        d_min=0.0,
        d_max=12.0,
        coord_scale=0.1,
    ):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.coord_scale = coord_scale
        self.rbf = RBFDistance(n_rbf=n_rbf, d_min=d_min, d_max=d_max)
        msg_scalar_in = hidden_dim * 2 + edge_dim + n_rbf + 1
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_scalar_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim * 2 + n_rbf + 1, edge_dim),
            nn.SiLU(), nn.Linear(edge_dim, edge_dim),
        )
        self.edge_norm = nn.LayerNorm(edge_dim)

    def _pair_geom(self, x_pos):
        diff = x_pos[:, :, None, :] - x_pos[:, None, :, :]
        d_raw = torch.norm(diff, dim=-1)
        rbf = self.rbf(d_raw)
        d_feat = (d_raw / self.d_max).clamp(max=1.0).unsqueeze(-1)
        return diff, d_raw, rbf, d_feat

    def forward(self, x_pos, x_att, e, atom_valid):
        B, M, D = x_att.shape
        eye = torch.eye(M, device=x_att.device, dtype=torch.bool).view(1, M, M)
        pair_valid = atom_valid[:, :, None] & atom_valid[:, None, :] & (~eye)
        denom = pair_valid.sum(dim=2, keepdim=True).clamp(min=1).float()

        diff, d_raw, rbf, d_feat = self._pair_geom(x_pos)
        x_i = x_att[:, :, None, :].expand(B, M, M, D)
        x_j = x_att[:, None, :, :].expand(B, M, M, D)
        msg_scalar = torch.cat([x_i, x_j, e, rbf, d_feat], dim=-1)
        m = self.msg_mlp(msg_scalar)
        gate = torch.sigmoid(self.gate_mlp(m))
        m = m * gate
        m = m.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        agg = m.sum(dim=2) / denom

        # NFU
        ns_in = torch.cat([x_att, agg], dim=-1)
        ds = self.node_mlp(ns_in)
        x_att = self.node_norm(x_att + ds)
        x_att = x_att.masked_fill(~atom_valid.unsqueeze(-1), 0.0)

        # NPU (reuses message m from above)
        coord_w = self.coord_mlp(m)
        coord_w = coord_w.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        unit = diff / (d_raw.unsqueeze(-1) + 1.0)
        dpos = (unit * coord_w).sum(dim=2) / denom
        x_pos = x_pos + self.coord_scale * dpos
        x_pos = x_pos.masked_fill(~atom_valid.unsqueeze(-1), 0.0)

        # EFU (recomputed on updated positions)
        _, _, rbf2, d_feat2 = self._pair_geom(x_pos)
        si = x_att[:, :, None, :].expand(B, M, M, D)
        sj = x_att[:, None, :, :].expand(B, M, M, D)
        edge_in = torch.cat([e, si, sj, rbf2, d_feat2], dim=-1)
        de = self.edge_mlp(edge_in)
        de = de.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        e = self.edge_norm(e + de)
        e = e.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        return x_pos, x_att, e


class VSGraphBlock(nn.Module):
    """One message-passing block: an optional V->S layer followed by an S->S layer."""
    def __init__(self, hidden_dim=256, k_hidden_dim=256, edge_dim=256,
                 n_rbf=48, d_max=12.0):
        super().__init__()
        # Kept as `k_to_m`/`m_to_m` (not `v_to_s`/`s_to_s`) so these
        # attributes' names -- and therefore their checkpoint state_dict
        # keys -- match already-trained checkpoints. `k_to_m` holds a
        # VtoSLayer, `m_to_m` holds a StoSLayer.
        self.k_to_m = VtoSLayer(
            a_hidden_dim=hidden_dim, k_hidden_dim=k_hidden_dim,
            edge_dim=edge_dim, hidden_dim=hidden_dim,
            n_rbf=n_rbf, d_max=d_max,
        )
        self.m_to_m = StoSLayer(
            hidden_dim=hidden_dim, edge_dim=edge_dim,
            n_rbf=n_rbf, d_max=d_max,
        )

    def forward(self, x_pos, x_att, e, v_pos, v_att, atom_valid, site_valid, use_v=True):
        if use_v:
            x_pos, x_att, e = self.k_to_m(
                x_pos=x_pos, x_att=x_att, e=e,
                v_pos=v_pos, v_att=v_att, atom_valid=atom_valid, site_valid=site_valid,
            )
        x_pos, x_att, e = self.m_to_m(
            x_pos=x_pos, x_att=x_att, e=e, atom_valid=atom_valid,
        )
        return x_pos, x_att, e


class BondOutputHead(nn.Module):
    def __init__(self, hidden_dim=256, edge_dim=256, n_rbf=48, d_min=0.0, d_max=12.0, num_bond_classes=5):
        super().__init__()
        self.d_max = d_max
        self.rbf = RBFDistance(n_rbf=n_rbf, d_min=d_min, d_max=d_max)
        pair_dim = hidden_dim * 2 + edge_dim + n_rbf + 1
        self.head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_bond_classes),
        )

    def forward(self, x_pos, x_att, e, atom_valid):
        B, M, D = x_att.shape
        x_i = x_att[:, :, None, :].expand(B, M, M, D)
        x_j = x_att[:, None, :, :].expand(B, M, M, D)
        d_raw = torch.cdist(x_pos, x_pos)
        rbf = self.rbf(d_raw)
        d_feat = (d_raw / self.d_max).clamp(max=1.0).unsqueeze(-1)
        pair_in = torch.cat([x_i, x_j, e, rbf, d_feat], dim=-1)
        bond_logits = self.head(pair_in)
        bond_logits = 0.5 * (bond_logits + bond_logits.transpose(1, 2))
        pair_valid = atom_valid[:, :, None] & atom_valid[:, None, :]
        bond_logits = bond_logits.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        return bond_logits


class COMPRESSDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        k_hidden_dim=256,
        edge_dim=256,
        time_dim=64,
        atom_emb_dim=64,
        bond_emb_dim=64,
        n_rbf=48,
        rbf_dmax=12.0,
        n_blocks=6,
        num_atom_types=13,     # 0..10 real, 11 = unused/reserved slot, 12 MASK
        num_bond_types=6,      # 0 none, 1..4 real, 5 MASK
        num_atom_classes=12,   # 0..10 real, 11 = unused/reserved slot
        num_bond_classes=5,    # 0..4
        k_blocks=3,             # leading blocks that use the V->S layer
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim=time_dim)
        self.atom_embedding = AtomStateEmbedding(
            hidden_dim=hidden_dim, atom_emb_dim=atom_emb_dim, time_dim=time_dim,
            num_atom_types=num_atom_types,
        )
        self.bond_embedding = BondStateEmbedding(
            hidden_dim=edge_dim, bond_emb_dim=bond_emb_dim, time_dim=time_dim,
            num_bond_types=num_bond_types, n_rbf=n_rbf, d_max=rbf_dmax,
        )
        # Kept as `k_encoder` (not `v_encoder`) for checkpoint compatibility;
        # holds a VEncoder instance (embeds the COMPRESS sites V).
        self.k_encoder = VEncoder(
            hidden_dim=k_hidden_dim, n_rbf=n_rbf, d_max=rbf_dmax,
        )
        self.blocks = nn.ModuleList([
            VSGraphBlock(
                hidden_dim=hidden_dim, k_hidden_dim=k_hidden_dim, edge_dim=edge_dim,
                n_rbf=n_rbf, d_max=rbf_dmax,
            )
            for _ in range(n_blocks)
        ])
        self.k_blocks = n_blocks if k_blocks is None else k_blocks
        # Attribute head predicts charge from scalar features; positions come
        # directly from the equivariantly updated x_pos, not from a Linear.
        self.attr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_atom_classes),
        )
        self.bond_head = BondOutputHead(
            hidden_dim=hidden_dim, edge_dim=edge_dim, n_rbf=n_rbf,
            num_bond_classes=num_bond_classes, d_max=rbf_dmax,
        )

    def forward(self, St, At, Bt, t, V, atom_valid, site_valid):
        B, M, _ = St.shape
        x_pos = St[:, :, :3]
        if t.numel() == 1:
            t = t.expand(B)
        t_emb = self.time_emb(t)
        x_att = self.atom_embedding(St=St, At=At, t_emb=t_emb)
        e = self.bond_embedding(St=St, Bt=Bt, t_emb=t_emb)
        v_pos, v_att = self.k_encoder(V=V, site_valid=site_valid)
        for i, block in enumerate(self.blocks):
            x_pos, x_att, e = block(
                x_pos=x_pos, x_att=x_att, e=e,
                v_pos=v_pos, v_att=v_att, atom_valid=atom_valid, site_valid=site_valid,
                use_v=(i < self.k_blocks),
            )
        pred_attr = self.attr_head(x_att)
        S1_pred = torch.cat([x_pos, pred_attr], dim=-1)
        atom_logits = self.atom_head(x_att)
        bond_logits = self.bond_head(x_pos=x_pos, x_att=x_att, e=e, atom_valid=atom_valid)
        S1_pred = S1_pred.masked_fill(~atom_valid.unsqueeze(-1), 0.0)
        atom_logits = atom_logits.masked_fill(~atom_valid.unsqueeze(-1), 0.0)
        return S1_pred, atom_logits, bond_logits
