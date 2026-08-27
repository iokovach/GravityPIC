#ifndef mass_DEPOSITION_K_H_
#define mass_DEPOSITION_K_H_

#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

/* This routine computes the mass density due to the particles using cloud-in-cell
 * deposition. The Fab rho is assumed to be node-centered.
 *
 * Arguments:
 *     particles : a pointer to the particle array-of-structs
 *     ns        : the stride length of particle struct (the size of the struct in number of reals)
 *     np        : the number of particles
 *     weights   : the particle weights
 *     mass      : the mass of this particle species
 *     rho       : a Fab that will contain the density on exit
 *     lo        : a pointer to the lo corner of this valid box for rho, in index space
 *     hi        : a pointer to the hi corner of this valid box for rho, in index space
 *     plo       : the real position of the left-hand corner of the problem domain
 *     dx        : the mesh spacing
 *     ng        : the number of ghost cells in rho
*/
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void deposit_cic (P const& p, amrex::ParticleReal wp, amrex::Real mass,
                  amrex::Array4<amrex::Real> const& rho,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
{
    // dxi = inverse cell spacing (1/dx, 1/dy, 1/dz)
    // compute inverse cell volume
    amrex::Real inv_vol = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
    // weight here is not CIC weight 
    // compute mass density 
    amrex::Real qp = wp * mass * inv_vol;

    // Convert the particle's physical (x,y,z) position into cell indices
    // (p.pos(0) - plo[0]) is the particle's distance from the domain's lower-left corner
    // multiplying by dxi[0] (=1/dx) converts that physical distance into a number of cells
    // So lx = 3.7 means the particle sits 70% of the way between cell 3 and cell 4 in x
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

    // identify index of lower corner of cell 
    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    // weights for upper corners
    amrex::Real wx_hi = lx - i;
    amrex::Real wy_hi = ly - j;
    amrex::Real wz_hi = lz - k;

    // weights for lower corners
    amrex::Real wx_lo = amrex::Real(1.0) - wx_hi;
    amrex::Real wy_lo = amrex::Real(1.0) - wy_hi;
    amrex::Real wz_lo = amrex::Real(1.0) - wz_hi;

    // CIC weighted assignment of density to grid sites in rho
    // AddNoRet is a method to write synchronously across threads
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j,   k,   0), wx_lo*wy_lo*wz_lo*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j,   k+1, 0), wx_lo*wy_lo*wz_hi*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j+1, k,   0), wx_lo*wy_hi*wz_lo*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j+1, k+1, 0), wx_lo*wy_hi*wz_hi*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j,   k,   0), wx_hi*wy_lo*wz_lo*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j,   k+1, 0), wx_hi*wy_lo*wz_hi*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j+1, k,   0), wx_hi*wy_hi*wz_lo*qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j+1, k+1, 0), wx_hi*wy_hi*wz_hi*qp);
}

#endif
