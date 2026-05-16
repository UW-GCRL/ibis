      subroutine drystress
c ---------------------------------------------------------------------
c
c common blocks
c
      include 'implicit.h'
c
      include 'compar.h'
      include 'comsoi.h'
      include 'comveg.h'
c
c local variables
c
      integer i, k    ! loop indices
c
      real stressfac, ! to calculate moisture stress factor 
     >     awc,       ! available water content (fraction)
     >     znorm,     ! normalizing factor
     >     zwilt      ! function of awc, =1 if awc = 1 (no stress)
c
c stressfac determines the 'strength' of the soil moisture
c stress on physiological processes
c
c strictly speaking, stresst* is multiplied to the vmax
c parameters used in the photosynthesis calculations
c
c stressfac determines the shape of the soil moisture response
c
      stressfac = -5.0
c
      znorm = 1.0 - exp(stressfac)
c
      do 100 i = 1, npoi
c
c initialize stress parameter
c
        stresstl(i) = 0.0
        stresstu(i) = 0.0
c
c fraction of soil water uptake in each layer
c
        do 110 k = 1, nsoilay
c
c plant available water content (fraction)
c
          awc = min (1.0, max (0.0,
     >              (wsoi(i,k)*(1 - wisoi(i,k))   - swilt(i,k)) /
     >              (sfield(i,k) - swilt(i,k))
     >              )         )
c
          zwilt = (1. - exp(stressfac * awc)) / znorm
c
c update for each layer
c
          stressl(i,k) = froot(k,1) * max (0.0, min (1.0, zwilt))
          stressu(i,k) = froot(k,2) * max (0.0, min (1.0, zwilt))
c
c integral over rooting profile
c
          stresstl(i) = stresstl(i) + stressl(i,k)
          stresstu(i) = stresstu(i) + stressu(i,k)
c
 110    continue
c
 100  continue
c
c return to main program
c
      return
      end
c scopy and const utility subroutines removed from this subset:
c they are not called by drystress. If needed, copy from utilities.f.
c
