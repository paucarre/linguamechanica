import math

import torch
from pytorch3d import transforms


class SE3:
    def exp(self, twist: torch.Tensor) -> torch.Tensor:
        self.raise_not_implemented_error("exp")

    def log(self, element: torch.Tensor) -> torch.Tensor:
        self.raise_not_implemented_error("log")

    def chain(
        self, left_element: torch.Tensor, right_element: torch.Tensor
    ) -> torch.Tensor:
        self.raise_not_implemented_error("chain")

    def invert(self, element: torch.Tensor) -> torch.Tensor:
        self.raise_not_implemented_error("invert")

    def identity(self, batch_size: int) -> torch.Tensor:
        self.raise_not_implemented_error("identity")

    def act_vector(self, element: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        self.raise_not_implemented_error("act_vector")

    def act_point(self, element: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        self.raise_not_implemented_error("act_point")

    def repeat(self, element: torch.Tensor, batch_size) -> torch.Tensor:
        self.raise_not_implemented_error("repeat")

    def raise_not_implemented_error(self, name):
        subclasses = [cls.__name__ for cls in SE3.__subclasses__()]
        raise NotImplementedError(
            f"'{name}' not implemented in SE3 class. Use its subclasses: {subclasses}"
        )


class ProjectiveMatrix(SE3):
    def exp(self, twist: torch.Tensor) -> torch.Tensor:
        return transforms.se3_exp_map(twist)

    def log(self, transformation: torch.Tensor) -> torch.Tensor:
        return transforms.se3_log_map(transformation)

    def chain(
        self, left_batched: torch.Tensor, right_batched: torch.Tensor
    ) -> torch.Tensor:
        return (
            transforms.Transform3d(matrix=right_batched)
            .compose(transforms.Transform3d(matrix=left_batched))
            .get_matrix()
        )

    def invert(self, matrix: torch.Tensor) -> torch.Tensor:
        return transforms.Transform3d(matrix=matrix).inverse().get_matrix()

    def identity(self, batch_size: int) -> torch.Tensor:
        return transforms.Transform3d(
            matrix=torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        ).get_matrix()

    def element_shape(self):
        return [4, 4]

    def repeat(self, element: torch.Tensor, batch_size: int) -> torch.Tensor:
        if len(element.shape) == 2:
            return element.unsqueeze(0).repeat(batch_size, 1, 1)
        elif len(element.shape) == 3:
            return element.repeat(batch_size, 1, 1)

    def act_vector(self, idq: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        # self.raise_not_implemented_error("act_vector")
        pass

    def act_point(self, idq: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        # self.raise_not_implemented_error("act_point")
        pass


class ImplicitDualQuaternion(SE3):
    """
    For details see http://www.neil.dantam.name/papers/dantam2020robust.pdf

    Implicit Dual Quaternions are represented using a tuple of
    rotational and translational components. The rotational
    component is a quaternion while the translation is a 3D
    vector of euclidean velocities.

    Given a Dual Quaternion: `h + (1/2) * v ⊗ hε`
    Where:
        `h` is a rotation quaternion
        `v` is a translation vector
        `⊗` is the quaternion operator
        `ε` is the dual number (ε²=0)

    `h` can be represented as follows:
    `h = xi + yj + zk + w` where `(i, j, k)` are the bivectors of an euclidean 3D Geometric Algebra.
    Finally, `h` can also be represented as `h = hv + hw`

    The Implicit Dual Quaternion is represented as a quaternion and a 3D vector.
    Specifically, as a rotation Quaternion `h` and a 3D translation vector `v`.
    Like Dual Quaternions, `h = hv + hw`, where `hv=(x, y, z)` and `hw = w`
    """

    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon

    def act_vector(self, idq, vector):
        return self.quat_mul(idq, self.vect_to_quat(vector))[:3]

    def chain(self, left_idq, right_idq):
        lh = self.extract_h(left_idq)
        rh = self.extract_h(right_idq)
        h = self.quat_mul(lh, rh)
        lv = self.extract_v(left_idq)
        rv = self.extract_v(right_idq)
        velocity_rotated = self.quat_mul(
            lh, self.quat_mul(self.vect_to_quat(rv), self.quat_conj(lh))
        )
        v = velocity_rotated + self.vect_to_quat(lv)
        # TODO: maybe normalize `h` just in case it numerically degrades?
        return torch.cat([h, v[:, :3]], 1)

    def invert(self, idq):
        # TODO: test this
        h_inv = self.quat_conj(self.extract_h(idq))
        v = self.extract_v(idq)
        v_inv = -self.quat_mul(
            h_inv, self.quat_mul(self.vect_to_quat(v), self.quat_conj(h_inv))
        )
        return torch.cat([h_inv, v_inv[:, :3]], 1)

    def identity(self, batch_size: int) -> torch.Tensor:
        zero_h = torch.zeros(batch_size, 4)
        zero_h[:, 3] = 1
        zero_v = torch.zeros(batch_size, 3)
        return torch.cat([zero_h, zero_v], 1)

    def exp(self, twist):
        coord_v = twist[:, :3]
        w = twist[:, 3:]
        phi = torch.norm(w, p=2, dim=1, keepdim=True)
        cos = phi.cos()
        sin = phi.sin()
        # TODO: it might be possible to only compute a subset of `phi_square`
        # and `phi_quartic`
        phi_square = phi * phi
        phi_quartic = phi_square * phi_square

        mu_r = torch.zeros_like(phi)
        mu_r_singularity = phi.abs() < self.epsilon
        mu_r[~mu_r_singularity] = sin[~mu_r_singularity] / phi[~mu_r_singularity]
        mu_r[mu_r_singularity] = (
            1.0
            - (phi_square[mu_r_singularity] / 6.0)
            + (phi_quartic[mu_r_singularity] / 120.0)
        )

        mu_d = torch.zeros_like(phi)
        mu_d_singularity = phi_square.abs() < self.epsilon
        mu_d[~mu_d_singularity] = (
            2.0 - (cos[~mu_d_singularity] * 2.0 * mu_r[~mu_d_singularity])
        ) / phi_square[~mu_d_singularity]
        mu_d[mu_d_singularity] = (
            (4.0 / 3.0)
            - ((4.0 * phi_square[mu_d_singularity]) / 15.0)
            + ((8.0 * phi_quartic[mu_d_singularity]) / 315.0)
        )

        gamma = (w * coord_v).sum(1).unsqueeze(1)
        h = torch.cat([mu_r * w, cos], 1)
        hv = self.extract_hv(h)
        cross = torch.cross(hv, coord_v)
        v = ((2.0 * mu_r) * cross) + (cos * 2.0 * mu_r * coord_v) + (mu_d * gamma * w)
        return torch.cat([h, v], 1)

    def log(self, implicit_dual_quaternion_batch):
        h = self.extract_h(implicit_dual_quaternion_batch)
        v = self.extract_v(implicit_dual_quaternion_batch)
        hv = self.extract_hv(h)
        s = torch.norm(hv, p=2, dim=1, keepdim=True)
        c = self.extract_hw(h)

        phi = torch.atan2(s, c)
        phi_square = phi * phi
        phi_fourth = phi_square * phi_square

        w = torch.zeros_like(hv)
        w_singularity = s.abs().squeeze() < self.epsilon
        w[~w_singularity] = phi[~w_singularity] * (
            hv[~w_singularity] / s[~w_singularity]
        )
        w[w_singularity, :] = (
            1.0
            + (phi_square[w_singularity, :] / 6.0)
            + ((7.0 / 360.0) * phi_fourth[w_singularity, :])
        ) * hv[w_singularity, :]

        mu_r = torch.zeros_like(phi)
        mu_r_singularity = s.abs() < self.epsilon
        mu_r[~mu_r_singularity] = (c[~mu_r_singularity] * phi[~mu_r_singularity]) / s[
            ~mu_r_singularity
        ]
        mu_r[mu_r_singularity] = (
            1.0
            - (phi_square[mu_r_singularity] / 3.0)
            - (phi_fourth[mu_r_singularity] / 45.0)
        )

        mu_d = torch.zeros_like(mu_r)
        mu_d_singularity = phi_square < self.epsilon
        mu_d[~mu_d_singularity] = (1.0 - mu_r[~mu_d_singularity]) / phi_square[
            ~mu_d_singularity
        ]
        mu_d[mu_d_singularity] = (
            (1.0 / 3.0)
            + (phi_square[mu_d_singularity] / 45.0)
            + ((2.0 * phi_fourth[mu_d_singularity]) / 945.0)
        )

        inner = ((v / 2.0) * w).sum(1).unsqueeze(1)
        log_v = (mu_d * inner * w) + (mu_r * (v / 2.0)) + torch.cross((v / 2.0), w)
        return torch.cat([log_v, w], 1)

    def element_shape(self):
        return [4 + 3]

    def repeat(self, element: torch.Tensor, batch_size: int) -> torch.Tensor:
        if len(element.shape) == 1:
            return element.unsqueeze(0).repeat(batch_size, 1)
        elif len(element.shape) == 2:
            return element.repeat(batch_size, 1)

    def vect_to_quat(self, vector):
        return torch.cat([vector, torch.zeros_like(vector[:, 0:1])], 1)

    def quat_mul(self, lh, rh):
        ai, bi, ci, di = 3, 0, 1, 2
        la, lb, lc, ld = lh[:, ai], lh[:, bi], lh[:, ci], lh[:, di]
        ra, rb, rc, rd = rh[:, ai], rh[:, bi], rh[:, ci], rh[:, di]
        h_a = (la * ra) - (lb * rb) - (lc * rc) - (ld * rd)
        h_b = (la * rb) + (lb * ra) + (lc * rd) - (ld * rc)
        h_c = (la * rc) - (lb * rd) + (lc * ra) + (ld * rb)
        h_d = (la * rd) + (lb * rc) - (lc * rb) + (ld * ra)
        return torch.cat(
            [h_b.unsqueeze(1), h_c.unsqueeze(1), h_d.unsqueeze(1), h_a.unsqueeze(1)], 1
        )

    def quat_conj(self, h):
        return torch.cat([-self.extract_hv(h), self.extract_hw(h)], 1)

    def extract_v(self, idq):
        return idq[:, 4:]

    def extract_h(self, idq):
        return idq[:, :4]

    def extract_hv(self, h):
        return h[:, :3]

    def extract_hw(self, h):
        return h[:, 3:4]


if __name__ == "__main__":
    test = torch.tensor(
        [[math.sin(math.pi / 2.0), 0, 0, math.cos(math.pi / 2.0), 0, 0, 0]],
        requires_grad=True,
    ).float()
    zero_h = test
    se3_log = se3.log(zero_h)
    loss = se3_log.sum()
    loss.retain_grad()
    se3_log.retain_grad()
    loss.backward()
    print(se3_log.grad)
