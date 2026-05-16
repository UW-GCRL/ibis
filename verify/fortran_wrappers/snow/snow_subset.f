      subroutine snowheat (tlsub, fhtop, sflo, xl, chl)
c ---------------------------------------------------------------------
c
c sets up call to tridia to solve implicit snow heat conduction,
c using snow temperatures in tsno (in comsno). adds an extra
c buried-lower-veg layer to the bottom of the snow with 
c conduction coefficient conbur/xl and heat capacity chl*xl
c

      include 'implicit.h'
c
      include 'compar.h'
      include 'comsno.h'
      include 'comsoi.h'
c
c Arguments
c
      real chl                   ! specific heat of lower veg per l/s area (supplied)
      
      real tlsub(npoi),          ! temperature of buried lower veg (supplied, returned)
     >     fhtop(npoi),          ! heat flux into top snow layer from atmos (supplied)
     >     sflo(npoi,nsnolay+2), ! downward heat flow across layer boundaries (returned)
     >     xl(npoi)              ! (lai(i,1)+sai(i,1))*fl(i), lower-veg density(supplied)
c
c Local variables
c
      integer k, i,              ! loop indices
     >        km1,               ! used to avoid layer 0
     >        kp1                ! used to avoid layer nsnolay+2
c
      real rimp,                 ! implicit fraction of the calculation (0 to 1)
     >     conbur,               ! conduction coeff of buried lower veg layer 
     >                           ! for unit density xl=(lai+sai)*fl, in w m-2 k-1
     >     hfake,                ! arbitrary small thickness to allow processing 
     >                           ! for zero snow. (doesn't use index since tridia
     >                           ! not set up for index.)
     >     rwork,                ! to compute matrix diagonals and right-hand side
     >     dt,                   ! '
     >     dti                   ! '
c
      real con(npoi,nsnolay+2),  ! conduction coefficents between layers
     >     temp(npoi,nsnolay+1), ! combined snow and buried-veg temperatures
     >     d1(npoi,nsnolay+1),   ! diagonals of tridiagonal systems of equations 
     >     d2(npoi,nsnolay+1),   ! '
     >     d3(npoi,nsnolay+1),   ! '
     >     rhs(npoi,nsnolay+1),  ! right-hand sides of systems of equations
     >     w1(npoi,nsnolay+1),   ! work array needed by tridia
     >     w2(npoi,nsnolay+1)    ! '
c
c conbur (for xl=1) is chosen to be equiv to 10 cm of snow
c
      data rimp, conbur, hfake /1.0, 2.0, .01/
c
c copy snow and buried-lower-veg temperatures into combined
c array temp
c
      call scopy (npoi*nsnolay, tsno,  temp             )
      call scopy (npoi,         tlsub, temp(1,nsnolay+1))
c
c set conduction coefficients between layers
c
      do 100 k=1,nsnolay+2
        if (k.eq.1) then
          call const (con(1,k), npoi, 0.0)
c
        else if (k.le.nsnolay) then
          rwork = 0.5 / consno
          do 102 i=1,npoi
            con(i,k) = 1. / (   max(hsno(i,k-1),hfake)*rwork
     >                        + max(hsno(i,k)  ,hfake)*rwork )
  102     continue
c
        else if (k.eq.nsnolay+1) then
          rwork = 0.5 / consno
          do 104 i=1,npoi
            con(i,k) = 1. / (   max(hsno(i,k-1),hfake)*rwork
     >                        + 0.5*xl(i)/conbur )
  104     continue
c
        else if (k.eq.nsnolay+2) then
          rwork = 0.5 / conbur
          do 106 i=1,npoi
            con(i,k) = 1. / (   xl(i)*rwork
     >                        + 0.5*hsoi(1) / consoi(i,1) )
  106     continue
        endif
  100 continue
c
c set matrix diagonals and right-hand side. for layer nsnolay+1
c (buried-lower-veg layer), use explicit contact with soil, and
c multiply eqn through by xl*chl/dtime to allow zero xl.
c
      do 200 k=1,nsnolay+1
        km1 = max (k-1,1)
        kp1 = min (k+1,nsnolay+1)
c
        if (k.le.nsnolay) then
          rwork = dtime /(rhos*cice)
          do 202 i=1,npoi
            dt = rwork / (max(hsno(i,k),hfake))
            d1(i,k) =    - dt*rimp* con(i,k)
            d2(i,k) = 1. + dt*rimp*(con(i,k)+con(i,k+1))
            d3(i,k) =    - dt*rimp* con(i,k+1)
c
            rhs(i,k) = temp(i,k) + dt
     >               * ( (1.-rimp)*con(i,k)  *(temp(i,km1)-temp(i,k))
     >                 + (1.-rimp)*con(i,k+1)*(temp(i,kp1)-temp(i,k)) )
  202     continue
c
          if (k.eq.1) then 
            rwork = dtime /(rhos*cice)
            do 204 i=1,npoi
              dt = rwork / (max(hsno(i,k),hfake))
              rhs(i,k) = rhs(i,k) + dt*fhtop(i)
  204       continue
          endif
c
        else if (k.eq.nsnolay+1) then
c
          rwork = chl / dtime
          do 206 i=1,npoi
            dti = xl(i)*rwork
            d1(i,k) =     -  rimp* con(i,k)
            d2(i,k) = dti +  rimp*(con(i,k)+con(i,k+1))
            d3(i,k) = 0.
            rhs(i,k) = dti*temp(i,k)
     >               + ( (1.-rimp)*con(i,k)*(temp(i,km1)-temp(i,k))
     >                 + con(i,k+1)*(tsoi(i,1)-(1.-rimp)*temp(i,k)) )
  206     continue
        endif
  200 continue
c
c solve the tridiagonal systems
c
      call tridia (npoi, npoi, nsnolay+1, d1,d2,d3, rhs, temp, w1,w2)
c
c deduce downward heat fluxes between layers
c
      call scopy (npoi, fhtop, sflo(1,1))
c
      do 400 k=1,nsnolay+1
        if (k.le.nsnolay) then
          rwork = rhos*cice/dtime
          do 402 i=1,npoi
            sflo(i,k+1) = sflo(i,k) - rwork*hsno(i,k)
     >                                *(temp(i,k)-tsno(i,k))
  402     continue
c
        else
          rwork = chl/dtime
          do 404 i=1,npoi
            sflo(i,k+1) = sflo(i,k)
     >                  - xl(i)*rwork*(temp(i,nsnolay+1)-tlsub(i))
  404     continue
        endif
  400 continue
c
c copy temperature solution to tsno and tlsub, but not for
c points with no snow
c
      do 500 k=1,nsnolay
        do 502 i=1,npoi
          if (fi(i).gt.0.) tsno(i,k) = temp(i,k) 
  502   continue
  500 continue
c
      do 510 i=1,npoi
        if (fi(i).gt.0.) tlsub(i) = temp(i,nsnolay+1)
  510 continue
c
      return
      end
      subroutine vadapt (hcur, tcur, htop, indp, np, nlay) 
c ---------------------------------------------------------------------
c
c re-adapt snow layer thicknesses, so top thickness
c equals hsnotop and other thicknesses are equal
c
c also adjusts profile of tracer field tcur so its vertical
c integral is conserved (eg, temperature)
c
      include 'implicit.h'
c
      include 'compar.h'
c
c Arguments
c
      integer np,            ! number of snow pts in current strip (supplied)
     >        nlay           ! # of layer
c
      integer indp(npoi)     ! index of snow pts in current strip (supplied)
c
      real htop              ! prescribed top layer thickness (supplied)
c
      real hcur(npoi,nlay),  ! layer thicknesses (supplied and returned)     
     >     tcur(npoi,nlay)   ! tracer field (supplied and returned)
c
c local variables
c
      integer i, j, k, ko    ! loop indices
c
      real dz, rwork
c
      real ht(npoi),         ! storing variable for zold        
     >     h1(npoi),         ! to compute new layer thickness
     >     za(npoi),         ! 
     >     zb(npoi),         ! 
     >     zheat(npoi)       !
c
      real hnew(npoi,nsnolay),    ! new layer thickness
     >     tnew(npoi,nsnolay),    ! new temperatures of layers
     >     zold(npoi,nsnolay+1)   ! distances from surface to old layer boundaries
c
c if no snow or seaice points in current 1d strip, return. note
c that the index is not used below (for cray vec and efficiency)
c except in the final loop setting the returned values
c
      if (np.eq.0) return
c
c set distances zold from surface to old layer boundaries
c
      call const (zold(1,1), npoi, 0.0)
c
      do 300 k=1,nlay
        do 302 i=1,npoi
          zold(i,k+1) = zold(i,k) + hcur(i,k)
  302   continue
  300 continue
c
c set new layer thicknesses hnew (tot thickness is unchanged).
c if total thickness is less than nlay*htop (which should be
c le hsnomin), make all new layers equal including
c top one, so other layers aren't so thin. use epsilon to 
c handle zero (snow) points
c
      call scopy (npoi, zold(1,nlay+1), ht)
c
      rwork = nlay*htop
      do 304 i=1,npoi
        if (ht(i).ge.rwork) then
          h1(i) = (ht(i)-htop)/(nlay-1)
        else
          h1(i) = max (ht(i)/nlay, epsilon)
        endif
  304 continue
c
      do 306 k=1,nlay
        do 308 i=1,npoi
          hnew(i,k) = h1(i)
  308   continue
  306 continue
c
      rwork = nlay*htop
      do 310 i=1,npoi
        if (ht(i).ge.rwork) hnew(i,1) = htop
  310 continue
c
c integrate old temperature profile (loop 410) over each
c new layer (loop 400), to get new field tnew
c
      call const (zb, npoi, 0.0)
c
      do 400 k=1,nlay
c
        do 402 i=1,npoi
          za(i) = zb(i)
          zb(i) = za(i) + hnew(i,k)
  402   continue
        call const (zheat, npoi, 0.0)
c
        do 410 ko=1,nlay
          do 412 i=1,npoi
            if (za(i).lt.zold(i,ko+1) .and. zb(i).gt.zold(i,ko)) then
              dz = min(zold(i,ko+1),zb(i)) - max(zold(i,ko),za(i))
              zheat(i) = zheat(i) + tcur(i,ko)*dz
            endif
  412     continue
  410   continue
c
        do 420 i=1,npoi
          tnew(i,k) = zheat(i) / hnew(i,k)
  420   continue
c
  400 continue
c
c use index for final copy to seaice or snow arrays, to avoid
c changing soil values (when called for seaice) and to avoid
c changing nominal snow values for no-snow points (when called
c for snow)
c
      do 500 k=1,nlay
        do 502 j=1,np
          i = indp(j)
          hcur(i,k) = hnew(i,k)
          tcur(i,k) = tnew(i,k)
  502   continue
  500 continue
c
      return
      end
      subroutine scopy (nt, arr, brr)
c ---------------------------------------------------------------------
c
c copies array arr to brr,for 1st nt words of arr
c
      include 'implicit.h'
c
c Arguments
c
      integer nt     
      real arr(nt),    ! input
     >     brr(nt)     ! output
c
c Local variables
c
      integer ia
c
      do 100 ia = 1, nt
        brr(ia) = arr(ia)
 100  continue
c
      return
      end
      subroutine const (arr, nar, value)
c ---------------------------------------------------------------------
c
c sets all elements of real vector arr to value
c
      include 'implicit.h'
c
c Arguments
c
      integer nar
c     
      real value
      real arr(nar)
c
c Local variables
c
      integer j
c
      do 100 j = 1, nar
        arr(j) = value
 100  continue
c
      return
      end
c
      subroutine tridia (ns, nd, ne, a, b, c, y, x, alpha, gamma)
c ---------------------------------------------------------------------
c
      include 'implicit.h'
c
      include 'compar.h'
c
c     purpose:
c     to compute the solution of many tridiagonal linear systems.
c
c      arguments:
c
c      ns ..... the number of systems to be solved.
c
c      nd ..... first dimension of arrays (ge ns).
c
c      ne ..... the number of unknowns in each system.
c               this must be > 2. second dimension of arrays.
c
c      a ...... the subdiagonals of the matrices are stored
c               in locations a(j,2) through a(j,ne).
c
c      b ...... the main diagonals of the matrices are stored
c               in locations b(j,1) through b(j,ne).
c
c      c ...... the super-diagonals of the matrices are stored in
c               locations c(j,1) through c(j,ne-1).
c
c      y ...... the right hand side of the equations is stored in
c               y(j,1) through y(j,ne).
c
c      x ...... the solutions of the systems are returned in
c               locations x(j,1) through x(j,ne).
c
c      alpha .. work array dimensioned alpha(nd,ne)
c
c      gamma .. work array dimensioned gamma(nd,ne)
c
c       history:  based on a streamlined version of the old ncar
c                 ulib subr trdi used in the phoenix climate
c                 model of schneider and thompson (j.g.r., 1981).
c                 revised by starley thompson to solve multiple
c                 systems and vectorize well on the cray-1.
c                 later revised to include a parameter statement
c                 to define loop limits and thus enable cray short
c                 vector loops.
c
c       algorithm:  lu decomposition followed by solution.
c                   note: this subr executes satisfactorily
c                   if the input matrix is diagonally dominant
c                   and non-singular.  the diagonal elements are
c                   used to pivot, and no tests are made to determine
c                   singularity. if a singular or numerically singular
c                   matrix is used as input a divide by zero or
c                   floating point overflow will result.
c
c       last revision date:      4 february 1988
c
c
c Arguments
c
      integer ns,     ! number of systems to be solved.
     >        nd,     ! first dimension of arrays (ge ns)
     >        ne      ! number of unknowns in each system. (>2)
      
      real 
     >  a(nd,ne),     ! subdiagonals of matrices stored in a(j,2)...a(j,ne).
     >  b(nd,ne),     ! main diagonals of matrices stored in b(j,1)...b(j,ne).
     >  c(nd,ne),     ! super-diagonals of matrices stored in c(j,1)...c(j,ne-1).
     >  y(nd,ne),     ! right hand side of equations stored in y(j,1)...y(j,ne).
     >  x(nd,ne),     ! solutions of the systems returned in x(j,1)...x(j,ne).
     >  alpha(nd,ne), ! work array 
     >  gamma(nd,ne)  ! work array
c
c local variables
c
      integer nm1,    !
     >  j, i, ib      ! loop indices

c
      nm1 = ne-1
c
c obtain the lu decompositions
c
      do 10 j=1,ns
         alpha(j,1) = 1./b(j,1)
         gamma(j,1) = c(j,1)*alpha(j,1)
   10 continue
      do 11 i=2,nm1
         do 12 j=1,ns
            alpha(j,i) = 1./(b(j,i)-a(j,i)*gamma(j,i-1))
            gamma(j,i) = c(j,i)*alpha(j,i)
   12    continue
   11 continue
c
c solve
c
      do 20 j=1,ns
         x(j,1) = y(j,1)*alpha(j,1)
   20 continue
      do 21 i=2,nm1
         do 22 j=1,ns
            x(j,i) = (y(j,i)-a(j,i)*x(j,i-1))*alpha(j,i)
   22    continue
   21 continue
      do 23 j=1,ns
         x(j,ne) = (y(j,ne)-a(j,ne)*x(j,nm1))/
     >             (b(j,ne)-a(j,ne)*gamma(j,nm1))
   23 continue
      do 24 i=1,nm1
         ib = ne-i
         do 25 j=1,ns
            x(j,ib) = x(j,ib)-gamma(j,ib)*x(j,ib+1)
   25    continue
   24 continue
c
      return
      end


c ---------------------------------------------------------------------
c snow_melt_block -- carved from snow.f lines 172-181.
c Reads hsno/tsno/fi from comsno; updates hsno/tsno; accumulates
c zmelt/zheat per point.
c ---------------------------------------------------------------------
      subroutine snow_melt_block(zmelt, zheat)
      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'

      real zmelt(npoi), zheat(npoi)

      integer i, k
      real rwork
      real xh(npoi), xm(npoi)

      rwork = 1.0 / rhos
      do 300 k = 1, nsnolay
        do 302 i = 1, npoi
          xh(i) = rhos*hsno(i,k)*cice * max(tsno(i,k)-tmelt, 0.)
          xm(i) = min(rhos*hsno(i,k), xh(i)/hfus)
          hsno(i,k) = hsno(i,k) - xm(i)*rwork
          tsno(i,k) = min(tsno(i,k), tmelt)
          zmelt(i) = zmelt(i) + fi(i)*xm(i)
          zheat(i) = zheat(i) + fi(i)*(xh(i)-hfus*xm(i))
  302   continue
  300 continue

      return
      end


c ---------------------------------------------------------------------
c snow_coverage_block -- carved from snow.f lines 187-210.
c
c Important divergence from JAX:
c   F77 here does NOT substitute hsnomin when ht=0.  JAX does
c   (`ht = where(ht < 1e-10, hsnomin, ht)`).  For physically valid
c   inputs (ht=0 iff fi=0), both produce finew=0.  Tests should not
c   exercise the degenerate (ht=0, fi>0) case.
c ---------------------------------------------------------------------
      subroutine snow_coverage_block(hsnoruf_in)
      include 'implicit.h'
      include 'compar.h'
      include 'comsno.h'

      real hsnoruf_in(npoi)

      integer i, k
      real zhh, zdh, finew
      real ht(npoi), x1(npoi)

      call const(ht, npoi, 0.0)
      do 400 k = 1, nsnolay
        do 402 i = 1, npoi
          ht(i) = ht(i) + hsno(i,k)
  402   continue
  400 continue

      zhh = 0.5 * hsnomin
      do 404 i = 1, npoi
        zdh = hsnoruf_in(i) - hsnomin
        finew = (-zhh + sqrt(zhh*zhh + zdh*fi(i)*ht(i))) / zdh
        finew = max(0., min(fimax, finew))
        x1(i) = fi(i) / max(finew, epsilon)
        fi(i) = finew
  404 continue

      do 406 k = 1, nsnolay
        do 408 i = 1, npoi
          hsno(i,k) = hsno(i,k) * x1(i)
  408   continue
  406 continue

      return
      end
