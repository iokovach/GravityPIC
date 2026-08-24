#ifndef INTERPOLATION_K_H_
#define INTERPOLATION_K_H_

#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

/* This routine interpolates the gravitational field to the particle positions
* using cloud-in-cell interpolation. The gravity field is assumed to be
* node-centered.
*
* Arguments:
*     particles : a pointer to the particle array-of-structs
*     ns        : the stride length of particle struct (the size of the struct in number of reals)
*     np        : the number of particles
*     Ex_p      : the field in the x-direction at the particle positions (output)
*     Ey_p      : the field in the y-direction at the particle positions (output)
*     Ez_p      : the field in the z-direction at the particle positions (output)
*     Ex, Ey, Ez: Fabs containing the field on the mesh
*     lo        : a pointer to the lo corner of this valid box, in index space
*     hi        : a pointer to the hi corner of this valid box, in index space
*     plo       : the real position of the left-hand corner of the problem domain
*     dx        : the mesh spacing
*     ng        : the number of ghost cells for the field
*/
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void interpolate_cic (P const& p, amrex::Real& Ex_p, amrex::Real& Ey_p, amrex::Real& Ez_p,
                      amrex::Array4<amrex::Real const> const& Ex,
                      amrex::Array4<amrex::Real const> const& Ey,
                      amrex::Array4<amrex::Real const> const& Ez,
                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
{
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    amrex::Real wx_hi = lx - i;
    amrex::Real wy_hi = ly - j;
    amrex::Real wz_hi = lz - k;

    amrex::Real wx_lo = amrex::Real(1.0) - wx_hi;
    amrex::Real wy_lo = amrex::Real(1.0) - wy_hi;
    amrex::Real wz_lo = amrex::Real(1.0) - wz_hi;

    Ex_p = wx_lo*wy_lo*wz_lo*Ex(i,   j,   k,   0) +
           wx_lo*wy_lo*wz_hi*Ex(i,   j,   k+1, 0) +
           wx_lo*wy_hi*wz_lo*Ex(i,   j+1, k,   0) +
           wx_lo*wy_hi*wz_hi*Ex(i,   j+1, k+1, 0) +
           wx_hi*wy_lo*wz_lo*Ex(i+1, j,   k,   0) +
           wx_hi*wy_lo*wz_hi*Ex(i+1, j,   k+1, 0) +
           wx_hi*wy_hi*wz_lo*Ex(i+1, j+1, k,   0) +
           wx_hi*wy_hi*wz_hi*Ex(i+1, j+1, k+1, 0);

    Ey_p = wx_lo*wy_lo*wz_lo*Ey(i,   j,   k,   0) +
           wx_lo*wy_lo*wz_hi*Ey(i,   j,   k+1, 0) +
           wx_lo*wy_hi*wz_lo*Ey(i,   j+1, k,   0) +
           wx_lo*wy_hi*wz_hi*Ey(i,   j+1, k+1, 0) +
           wx_hi*wy_lo*wz_lo*Ey(i+1, j,   k,   0) +
           wx_hi*wy_lo*wz_hi*Ey(i+1, j,   k+1, 0) +
           wx_hi*wy_hi*wz_lo*Ey(i+1, j+1, k,   0) +
           wx_hi*wy_hi*wz_hi*Ey(i+1, j+1, k+1, 0);

    Ez_p = wx_lo*wy_lo*wz_lo*Ez(i,   j,   k,   0) +
           wx_lo*wy_lo*wz_hi*Ez(i,   j,   k+1, 0) +
           wx_lo*wy_hi*wz_lo*Ez(i,   j+1, k,   0) +
           wx_lo*wy_hi*wz_hi*Ez(i,   j+1, k+1, 0) +
           wx_hi*wy_lo*wz_lo*Ez(i+1, j,   k,   0) +
           wx_hi*wy_lo*wz_hi*Ez(i+1, j,   k+1, 0) +
           wx_hi*wy_hi*wz_lo*Ez(i+1, j+1, k,   0) +
           wx_hi*wy_hi*wz_hi*Ez(i+1, j+1, k+1, 0);
}

// interpolator for the case where we have a coarse grid and a finer grid
// for another day -- if i need more than 2 grid widths then i might need to mod this
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void interpolate_cic_two_levels (P const& p, amrex::Real& Ex_p, amrex::Real& Ey_p, amrex::Real& Ez_p,
                                 amrex::Array4<amrex::Real const> const& Ex,
                                 amrex::Array4<amrex::Real const> const& Ey,
                                 amrex::Array4<amrex::Real const> const& Ez,
                                 amrex::Array4<amrex::Real const> const& cEx,
                                 amrex::Array4<amrex::Real const> const& cEy,
                                 amrex::Array4<amrex::Real const> const& cEz,
                                 amrex::Array4<int const> const& mask,
                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& cdxi,
                                 int lev)
{
    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    if (lev == 1 && mask(i,j,k,0) == 1) {
        // use coarse fields
        lx = (p.pos(0) - plo[0]) * cdxi[0];
        ly = (p.pos(1) - plo[1]) * cdxi[1];
        lz = (p.pos(2) - plo[2]) * cdxi[2];

        i = static_cast<int>(amrex::Math::floor(lx));
        j = static_cast<int>(amrex::Math::floor(ly));
        k = static_cast<int>(amrex::Math::floor(lz));

        amrex::Real wx_hi = lx - i;
        amrex::Real wy_hi = ly - j;
        amrex::Real wz_hi = lz - k;

        amrex::Real wx_lo = amrex::Real(1.0) - wx_hi;
        amrex::Real wy_lo = amrex::Real(1.0) - wy_hi;
        amrex::Real wz_lo = amrex::Real(1.0) - wz_hi;

        Ex_p = wx_lo*wy_lo*wz_lo*cEx(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*cEx(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*cEx(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*cEx(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*cEx(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*cEx(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*cEx(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*cEx(i+1, j+1, k+1, 0);

        Ey_p = wx_lo*wy_lo*wz_lo*cEy(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*cEy(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*cEy(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*cEy(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*cEy(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*cEy(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*cEy(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*cEy(i+1, j+1, k+1, 0);

        Ez_p = wx_lo*wy_lo*wz_lo*cEz(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*cEz(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*cEz(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*cEz(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*cEz(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*cEz(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*cEz(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*cEz(i+1, j+1, k+1, 0);
    } else {
        // use fine fields
        amrex::Real wx_hi = lx - i;
        amrex::Real wy_hi = ly - j;
        amrex::Real wz_hi = lz - k;

        amrex::Real wx_lo = amrex::Real(1.0) - wx_hi;
        amrex::Real wy_lo = amrex::Real(1.0) - wy_hi;
        amrex::Real wz_lo = amrex::Real(1.0) - wz_hi;

        Ex_p = wx_lo*wy_lo*wz_lo*Ex(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*Ex(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*Ex(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*Ex(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*Ex(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*Ex(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*Ex(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*Ex(i+1, j+1, k+1, 0);

        Ey_p = wx_lo*wy_lo*wz_lo*Ey(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*Ey(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*Ey(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*Ey(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*Ey(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*Ey(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*Ey(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*Ey(i+1, j+1, k+1, 0);

        Ez_p = wx_lo*wy_lo*wz_lo*Ez(i,   j,   k,   0) +
               wx_lo*wy_lo*wz_hi*Ez(i,   j,   k+1, 0) +
               wx_lo*wy_hi*wz_lo*Ez(i,   j+1, k,   0) +
               wx_lo*wy_hi*wz_hi*Ez(i,   j+1, k+1, 0) +
               wx_hi*wy_lo*wz_lo*Ez(i+1, j,   k,   0) +
               wx_hi*wy_lo*wz_hi*Ez(i+1, j,   k+1, 0) +
               wx_hi*wy_hi*wz_lo*Ez(i+1, j+1, k,   0) +
               wx_hi*wy_hi*wz_hi*Ez(i+1, j+1, k+1, 0);
    }
}

#endif