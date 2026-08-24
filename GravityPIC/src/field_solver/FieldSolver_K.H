#ifndef FIELD_SOLVER_K_H_
#define FIELD_SOLVER_K_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

// This averages a fine-level node-centered field down onto the corresponding coarse-level node
// Each coarse node (i,j,k) has a fine node (ii,jj,kk) at the same physical location, plus a weighted
// average of the fine nodes surrounding it 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void sum_fine_to_crse_nodal (int i, int j, int k,
                             amrex::Array4<amrex::Real> const& crse,
                             amrex::Array4<amrex::Real const> const& fine,
                             amrex::IntVect const& ratio) noexcept
{
    const int facx = ratio[0];
    const int facy = ratio[1];
    const int facz = ratio[2];
    const int ii = i*facx;
    const int jj = j*facy;
    const int kk = k*facz;
    // weight of each node is about how many coarse nodes that node is shared between — not about how many nodes are added!
    crse(i,j,k,0) =
        // center node, weight 1
        fine(ii,jj,kk,0)
        // 6 face neighbors weight 1/2
        + amrex::Real(0.5)*(fine(ii-1,jj,  kk,  0) + fine(ii+1,jj,  kk,  0) +
                            fine(ii,  jj-1,kk,  0) + fine(ii,  jj+1,kk,  0) +
                            fine(ii,  jj,  kk-1,0) + fine(ii,  jj,  kk+1,0))
        // 12 edge neighbors weight 1/4
        + amrex::Real(0.25)*(fine(ii-1,jj-1,kk,  0) + fine(ii-1,jj+1,kk,  0) +
                             fine(ii+1,jj-1,kk,  0) + fine(ii+1,jj+1,kk,  0) +
                             fine(ii-1,jj,  kk-1,0) + fine(ii-1,jj,  kk+1,0) +
                             fine(ii+1,jj,  kk-1,0) + fine(ii+1,jj,  kk+1,0) +
                             fine(ii,  jj-1,kk-1,0) + fine(ii,  jj-1,kk+1,0) +
                             fine(ii,  jj+1,kk-1,0) + fine(ii,  jj+1,kk+1,0))
        // 8 corner neighbors weight 1/8
        + amrex::Real(0.125)*(fine(ii-1,jj-1,kk-1,0) + fine(ii-1,jj-1,kk+1,0) +
                              fine(ii-1,jj+1,kk-1,0) + fine(ii-1,jj+1,kk+1,0) +
                              fine(ii+1,jj-1,kk-1,0) + fine(ii+1,jj-1,kk+1,0) +
                              fine(ii+1,jj+1,kk-1,0) + fine(ii+1,jj+1,kk+1,0));

    // Normalize by the total stencil weight (8 in 3D, was 4 in 2D).
    crse(i,j,k,0) = crse(i,j,k,0) / amrex::Real(8.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void zero_out_bndry (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& input,
                     amrex::Array4<amrex::Real> const& bndry,
                     amrex::Array4<int const> mask) noexcept
{

    if (mask(i,j,k,0) == 1) {
        bndry(i,j,k,0) = input(i,j,k,0);
        input(i,j,k,0) = amrex::Real(0.0);
    }
}

/**
   tmp_mask stores 1 if a ghost cell is either uncovered or outside the
   physical domain.
   On exit, mask will be 1 for cells with ncells of either of those conditions,
   zero otherwise.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void build_mask (int i, int j, int k,
                 amrex::Array4<int const> tmp_mask,
                 amrex::Array4<int> mask, int ncells) noexcept
{
    int total = 0;
    for (int kk = k-ncells; kk <= k+ncells; ++kk) {
        for (int jj = j-ncells; jj <= j+ncells; ++jj) {
            for (int ii = i-ncells; ii <= i+ncells; ++ii) {
                total += tmp_mask(ii, jj, kk, 0);
            }
        }
    }
    mask(i, j, k, 0) = (total > 0) ? 1 : 0;
}

/**
 * \brief This routine computes the node-centered field given a node-centered phi.
 * The gradient is computed using 2nd-order centered differences. It assumes the
 * boundary conditions have already been set and that you have one row of ghost cells.
 * Note that this routine includes the minus sign in E = - grad phi.
 *
 * Arguments:
 *     lo, hi:       The corners of the valid box over which the gradient is taken
 *     Ex, Ey, Ez:   The field in the x, y, and z directions.
 *     dx:           The cell spacing
 *
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_E_nodal (int i, int j, int k,
                      amrex::Array4<amrex::Real> const& Ex,
                      amrex::Array4<amrex::Real> const& Ey,
                      amrex::Array4<amrex::Real> const& Ez,
                      amrex::Array4<amrex::Real const> const& phi,
                      amrex::GpuArray<amrex::Real, 3> dx) noexcept
{
    // Centered-difference gradient: d(phi)/dx (i,j,k) ~= (phi(i-1) - phi(i+1)) / (2*dx),
    // with the minus sign from E = -grad(phi) already folded into the ordering
    amrex::GpuArray<amrex::Real, 3> fac;
    fac[0] = amrex::Real(0.5) / dx[0];
    fac[1] = amrex::Real(0.5) / dx[1];
    fac[2] = amrex::Real(0.5) / dx[2];

    Ex(i,j,k,0) = fac[0] * (phi(i-1,j,k,0) - phi(i+1,j,k,0));
    Ey(i,j,k,0) = fac[1] * (phi(i,j-1,k,0) - phi(i,j+1,k,0));
    Ez(i,j,k,0) = fac[2] * (phi(i,j,k-1,0) - phi(i,j,k+1,0));
};

#endif