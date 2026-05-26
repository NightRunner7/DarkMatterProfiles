(* ::Package:: *)

Off[General::stop]

hydroeq = 
  Compile[{{\[Rho], _Real, 1}, {r, _Real, 1}, {\[Sigma], _Real, 
     1}, {M, _Real, 1}, {tG, _Real}, {t0, _Real}}, 
   Module[{n = Length@\[Rho], 
     inv = \[Rho], \[Phi] = \[Rho], \[Psi] = \[Rho], a\[Phi] = \[Rho],
      b\[Phi] = \[Rho], c\[Phi] = \[Rho], d\[Phi] = \[Rho], 
     a\[Psi] = \[Rho], b\[Psi] = \[Rho], c\[Psi] = \[Rho], 
     d\[Psi] = \[Rho], \[Alpha] = \[Rho], \[Gamma] = \[Rho], \
\[Delta]r = \[Rho], \[Delta]\[Rho] = \[Rho], n\[Rho] = \[Rho], 
     n\[Sigma] = \[Sigma], nr = r},
    Do[
     Do[inv[[j]] = (n\[Sigma][[j]])^3/n\[Rho][[j]], {j, 1, n}];
     Do[\[Phi][[
        j]] = (n\[Rho][[j]]^(5/3) inv[[j]]^(2/3) - 
           n\[Rho][[j - 1]]^(5/3) inv[[j - 1]]^(2/3))/(M[[j]] - 
           M[[j - 1]]) + ((t0/tG)^2 (M[[j]] + 
             M[[j - 1]]))/(2 \[Pi]^2 (nr[[j]] + nr[[j - 1]])^4), {j, 
       2, n}];
     Do[\[Psi][[j]] = (M[[j]] - M[[j - 1]])/(nr[[j]] - nr[[j - 1]]) - 
        1/2 \[Pi] (nr[[j]] + nr[[j - 1]])^2 (n\[Rho][[j]] + 
           n\[Rho][[j - 1]]), {j, 2, n}];
     (**)
     Do[a\[Phi][[
        j]] = -((2 (t0/tG)^2 (M[[j]] + 
              M[[j - 1]]))/(\[Pi]^2 (nr[[j]] + nr[[j - 1]])^5)), {j, 
       2, n}];
     Do[b\[Phi][[
        j]] = -((5 (n\[Rho][[j - 1]]^(2/3) inv[[
                j - 1]]^(2/3)))/(3 (M[[j]] - M[[j - 1]]))), {j, 2, n}];
     Do[c\[Phi][[
        j]] = -((2 (t0/tG)^2 (M[[j]] + 
              M[[j - 1]]))/(\[Pi]^2 (nr[[j]] + nr[[j - 1]])^5)), {j, 
       2, n}];
     Do[d\[Phi][[
        j]] = (5 (n\[Rho][[j]]^(2/3) inv[[j]]^(2/3)))/(3 (M[[j]] - 
            M[[j - 1]])), {j, 2, n}];
     Do[a\[Psi][[
        j]] = (M[[j]] - 
           M[[j - 1]])/(nr[[j]] - nr[[j - 1]])^2 - \[Pi] (nr[[j]] + 
           nr[[j - 1]]) (n\[Rho][[j]] + n\[Rho][[j - 1]]), {j, 2, n}];
     Do[b\[Psi][[j]] = 1/2 (-\[Pi]) (nr[[j]] + nr[[j - 1]])^2, {j, 2, 
       n}];
     Do[c\[Psi][[
        j]] = -((M[[j]] - 
             M[[j - 1]])/(nr[[j]] - nr[[j - 1]])^2) - \[Pi] (nr[[j]] +
            nr[[j - 1]]) (n\[Rho][[j]] + n\[Rho][[j - 1]]), {j, 2, n}];
     Do[d\[Psi][[j]] = 1/2 (-\[Pi]) (nr[[j]] + nr[[j - 1]])^2, {j, 2, 
       n}];
     (**)
     \[Alpha][[1]] = 0;
     Do[\[Alpha][[
        j]] = (d\[Phi][[
            j]] (b\[Psi][[j]] - a\[Psi][[j]] \[Alpha][[j - 1]]) - 
          d\[Psi][[
            j]] (b\[Phi][[j]] - 
             a\[Phi][[j]] \[Alpha][[j - 1]]))/(c\[Phi][[
            j]] (b\[Psi][[j]] - a\[Psi][[j]] \[Alpha][[j - 1]]) - 
          c\[Psi][[
            j]] (b\[Phi][[j]] - a\[Phi][[j]] \[Alpha][[j - 1]])), {j, 
       2, n}];
     \[Gamma][[1]] = 0;
     Do[\[Gamma][[
        j]] = ((b\[Psi][[j]] - 
             a\[Psi][[j]] \[Alpha][[j - 1]]) (\[Phi][[j]] - 
             a\[Phi][[j]] \[Gamma][[j - 1]]) - (b\[Phi][[j]] - 
             a\[Phi][[j]] \[Alpha][[j - 1]]) (\[Psi][[j]] - 
             a\[Psi][[j]] \[Gamma][[j - 1]]))/(c\[Phi][[
            j]] (b\[Psi][[j]] - a\[Psi][[j]] \[Alpha][[j - 1]]) - 
          c\[Psi][[
            j]] (b\[Phi][[j]] - a\[Phi][[j]] \[Alpha][[j - 1]])), {j, 
       2, n}];
     (**)
     \[Delta]\[Rho][[n]] = 0;
     \[Delta]r[[n]] = -\[Gamma][[n]];
     Do[\[Delta]\[Rho][[
        n - j]] = -((\[Psi][[n - j + 1]] - 
            a\[Psi][[n - j + 1]] \[Gamma][[n - j]] + 
            c\[Psi][[n - j + 1]] \[Delta]r[[n - j + 1]] + 
            
            d\[Psi][[n - j + 1]] \[Delta]\[Rho][[
              n - j + 1]])/(b\[Psi][[n - j + 1]] - 
            a\[Psi][[n - j + 1]] \[Alpha][[n - j]]));
      \[Delta]r[[
        n - j]] = -\[Gamma][[n - j]] - \[Alpha][[
          n - j]]*\[Delta]\[Rho][[n - j]];, {j, 1, n - 1}];
     (**)
     If[Max[Table[Abs[\[Delta]r[[j]]/r[[j]]], {j, 1, n}]] < 
       10^-4.(*corvergence criterion*), Break[]];
     Do[nr[[j]] = nr[[j]] + \[Delta]r[[j]], {j, 1, n}];
     Do[n\[Rho][[j]] = n\[Rho][[j]] + \[Delta]\[Rho][[j]], {j, 1, n}];
     Do[n\[Sigma][[j]] = (n\[Rho][[j]]*inv[[j]])^(1/3), {j, 1, n}];
     , {k, 0, 10^10}];
    (**)
    {nr, n\[Rho], n\[Sigma]}
    ]
   , CompilationTarget -> "C", RuntimeOptions -> "Speed"];
(**)
heat = Compile[{{\[Rho], _Real, 1}, {r, _Real, 1}, {\[Sigma], _Real, 
     1}, {M, _Real, 1}, {\[Xi], _Real, 
     1}, {tG, _Real}, {t0, _Real}, {m, _Real}, {r0, _Real}, {\[Rho]0, \
_Real}},
   Module[{n = Length@\[Rho], n\[Rho] = \[Rho], n\[Sigma] = \[Sigma], 
     nr = r, n\[Xi] = \[Xi], \[Kappa] = r, 
     L = r, \[Delta]ucond = r, \[Delta]usemi = r, \[Delta]u = 
      r, \[Delta]T = r},
    Do[\[Kappa][[
       j]] = ((n\[Sigma][[j]]*(3/2 *(25 (\[Pi])^(1/2))/32))^-1 + (((
           2.18428*10^-6*m*t0)/(r0*n\[Xi][[j]]*\[Rho]0))/
          tG)^2*(3/2*0.75*(16/\[Pi])^(
           1/2)* (n\[Rho][[j]]*n\[Sigma][[j]]^3))^-1)^-1, {j, 1, n}];
    L[[1]] = 0;
    Do[L[[j]] = ((2.18428*10^-6*m*t0)/(r0*n\[Xi][[j]]*\[Rho]0))(*tel*)/
       t0*4 \[Pi] nr[[j]]^2*\[Kappa][[j]]*
       Piecewise[{{(n\[Sigma][[j + 1]]^2 - n\[Sigma][[j - 1]]^2)/(
          nr[[j + 1]] - nr[[j - 1]]), 
          1 < j < n}, {(-n\[Sigma][[j + 2]]^2 + 
           4 n\[Sigma][[j + 1]]^2 - 3 n\[Sigma][[j]]^2)/(
          nr[[j + 2]] - nr[[j]]), j == 1}, {(
          3 n\[Sigma][[j]]^2 - 4 n\[Sigma][[j - 1]]^2 + 
           n\[Sigma][[j - 2]]^2)/(nr[[j]] - nr[[j - 2]]), 
          j == n}}], {j, 2, n}];
    Do[\[Delta]ucond[[j]] = 
      Piecewise[{{(L[[j + 1]] - L[[j - 1]])/(M[[j + 1]] - M[[j - 1]]),
          1 < j < n}, {(-L[[j + 2]] + 4 L[[j + 1]] - 3 L[[j]])/(
         M[[j + 2]] - M[[j]]), j == 1}, {(
         3 L[[j]] - 4 L[[j - 1]] + L[[j - 2]])/(M[[j]] - M[[j - 2]]), 
         j == n}}], {j, 1, n}];
    Do[\[Delta]u[[
       j]] = (\[Delta]ucond[[j]])/(3/2 n\[Sigma][[j]]^2), {j, 1, n}];
    \[Delta]T[[
      1]] = (Max[
        Table[Abs[\[Delta]u[[j]]], {j, 1, 
          n}]])^-1*10^-4.(*determination of the next time step*);
    Do[n\[Sigma][[
       j]] = (Sqrt[
        2/3*((3/2 n\[Sigma][[j]]^2) + (\[Delta]ucond[[j]])*\[Delta]T[[
             1]])]), {j, 1, n}];
    {\[Delta]T, n\[Sigma]}
    ]
   , CompilationTarget -> "C", RuntimeOptions -> "Speed"];
(**)
(*concentration-mass relation*)
\[Rho]c = 1.8791*(h^2)*1.4771*10^-7(*Solar Mass/(pc)^3*);
h = 0.671;
a[z_] := 0.520 + (0.905 - 0.520)*Exp[-0.617*z^1.21];
b[z_] := -0.101 + 0.026*z;
c[M_, z_] := 
  10^a[z]*(M/((10^12)*(h^-1)(*Solar Mass*)))^
    b[z]*(13.3199/13.3199)(*5.26*(M/10^14)^-0.1*);
r0[M_, z_] := (200*4/3 \[Pi]*\[Rho]c)^(-1/3)*
   M^(1/3)*(c[M, z])^-1*10^-3(*in kpc*);
\[Rho]0[M_, z_] := 
  M/(4 \[Pi]*(r0[M, z]*10^3)^3) (1/(-(c[M, z]/(1 + c[M, z])) + 
       Log[1 + c[M, 
          z]]))(*\[Rho]c/3*c[M,z]^3*(-(c[M,z]/(1+c[M,z]))+Log[1+c[M,z]\
])^-1*)(*in Solar Mass/pc^3*);
z = 0;
Mi = GALAXYMASS;(*match the range in "Evolution"*)
Mf = 10.9;
NM = 1;
\[Delta]M = (Mf - Mi)/NM;
Do[M200[i] = 10^(Mi + \[Delta]M*i), {i, 0, NM}];
Do[RS[i] = r0[M200[i], 0], {i, 0, NM}]
Do[DS[i] = \[Rho]0[M200[i], 0], {i, 0, NM}]
Table[M200[i], {i, 0, NM}]
Table[RS[i], {i, 0, NM}]
Table[DS[i], {i, 0, NM}]
Clear[Mi, Mf, NM, r0, \[Rho]0, z, a, b, c, h, \[Rho]c, M];

SetDirectory[NotebookDirectory[]];(*output directory*)
(**)
Mi = GALAXYMASS;(*to scan halo masses, change also "p" range \
below*)(*match the range in "Compile functions"*)
Mf = 10.9;
NM = 1;
\[Delta]M = (Mf - Mi)/NM;
Do[(*mass loop*)
 Print["M=" <> ToString[Mi + \[Delta]M*p]];
 (*Set up initial r-grid in Log-bin*)
 ri = -2.(*Log10[Subscript[r, i=1]/Subscript[r, 0]]*);(*-4 for rSIDM*)

 
 rf = 2.(*Log10[Subscript[r, i=NR]/Subscript[r, 0]]*);(*2 for rSIDM*)

 
 NR = 400;(*700 for rSIDM*)
 \[Delta]R = (rf - ri)/NR;
 (*NFW scale parameters for given halo mass*)
 rs = RS[p](*in kpc*);
 \[Rho]s = DS[p](*in Solarmass/pc^3*);
 (**)
 G = 1/(1.220890 10^19)^2(*GeV^-2*);
 m = 1(*GeV*)(*dummy variable: do not change this!*);
 r0 = rs*1.56 10^35(*in GeV^-1,1 kpc*);
 t0 = 4.79 10^40(*in GeV^-1*)(*1 Gyr*);
 \[Rho]0 = (\[Rho]s)*2.94*10^-40(*Gev^4,Solarmass/pc^3*);(*Subscript[\
\[Sigma], 0] = Subscript[r, 0]/Subscript[t, 0]*)
 tG = Sqrt[1/(4 \[Pi] G \[Rho]0)];
 (**)
 t = 0.;
 Tticker = 0;
 ticker = 1;
 (*set r-grid*)
 r[0] = 0;
 Do[r[j] = 10^(ri + \[Delta]R*(j - 1)), {j, 1, NR + 1}];
 Table[r[j], {j, 0, NR + 1}];
 (*initial \[Rho]-grid*)
 \[Rho][0] = 
  NIntegrate[(4 \[Pi]*x^2)/(
    x*(1 + x)^2), {x, 0, 10^ri}]*(4/3 \[Pi]*10^(
     3*ri))^-1;(*assumed NFW*)
 Do[\[Rho][j] = 
   1/((10^(ri + \[Delta]R*(j - 1)))*(1 + 10^(
      ri + \[Delta]R*(j - 1)))^2), {j, 1, NR + 1}];
 Table[\[Rho][j], {j, 0, NR + 1}];
 (*initial M-grid*)
 M[0] = 0;
 M[1] = NIntegrate[(4 \[Pi]*x^2)/(
   x*(1 + x)^2), {x, 0, 10^ri}];(*assumed NFW*)
 Do[M[j] = 
   M[j - 1] + \[Pi]/
     2*(r[j] + r[j - 1])^2*(\[Rho][j] + \[Rho][j - 1])*(r[j] - 
       r[j - 1]), {j, 2, NR + 1}];
 Table[M[j], {j, 0, NR + 1}];
 (*initial \[Sigma]-grid*)
 \[Rho]\[Sigma][
   NR + 1] = \[Rho][
    NR + 1]*(16.9195*(2*\[Rho]s)^(
     1/2))^2;(*precisely \[Rho]\[Sigma]^2*)(*arbitrary as long as \
numerically stable*)
 Do[\[Rho]\[Sigma][(NR + 1) - 
     j] = \[Rho]\[Sigma][(NR + 1) - j + 1] + (t0/
      tG)^2*((M[(NR + 1) - j] + 
        M[(NR + 1) - j + 1])*(\[Rho][(NR + 1) - j] + \[Rho][(NR + 1) -
           j + 1]))/(
     4 \[Pi]*(r[(NR + 1) - j] + 
        r[(NR + 1) - j + 1])^2)*(r[(NR + 1) - j + 1] - 
       r[(NR + 1) - j]), {j, 1, NR + 1}];
 Do[\[Sigma][j] = Sqrt[\[Rho]\[Sigma][j]/\[Rho][j]], {j, 0, NR + 1}];
 Table[\[Sigma][j], {j, 0, NR + 1}];
 Table[{r[j], \[Sigma][j]}, {j, 0, NR + 1}];
 ListLogPlot[Table[{r[j], \[Sigma][j]}, {j, 0, NR + 1}]];
 (**)
 \[Rho]array = Table[\[Rho][j], {j, 0, NR + 1}];
 rarray = Table[r[j], {j, 0, NR + 1}];
 \[Sigma]harray = 
  Table[\[Sigma][j], {j, 0, NR + 1}](*not \[Sigma]array!*);
 Marray = Table[M[j], {j, 0, NR + 1}];
 Clear[\[Rho], r, \[Sigma]];
 (**)
 neweq = hydroeq[\[Rho]array, rarray, \[Sigma]harray, Marray, tG, t0];
 rarray = neweq[[1]];
 \[Rho]array = neweq[[2]];
 \[Sigma]array = neweq[[3]];
 
 Do[(*time loop*)
  (*Update SIDM cross section distribution with respect to the new \
equilibrium profile*)
  \[Xi]array = 
   Table[(*\[Xi]=*)
    CROSSSECTION(*10^\[Sigma]mRSIDMint[Log10[\[Sigma]array[[j]]*r0/
     t0]]*)/(1(*cm^2/g*))*0.01, {j, 1, 
     NR + 2}];(*velocity dependence taken into account*)
  (*Heat velocity dispersion.*)
  h = heat[\[Rho]array, rarray, \[Sigma]array, Marray, \[Xi]array, 
    tG,(*tJ,*)t0, m, r0, \[Rho]0];
  \[Delta]T = h[[1]][[1]](*time evolution step*);
  \[Sigma]harray = h[[2]](*heated velocity dispersion*);
  t = t + \[Delta]T;
  (*Find new hydrostatic equilibrium.*)
  neweq = hydroeq[\[Rho]array, rarray, \[Sigma]harray, Marray, tG, t0];
  rarray = neweq[[1]];
  \[Rho]array = neweq[[2]];
  \[Sigma]array = neweq[[3]];
  (*Record solution for the specified time slice, and print Kn*)
  If[
   Log10[t] >= -4.(*initial time when you wish to start recording, 
      Log10[Subscript[t, i]/t0]*)+ 
      0.0025(*time resolution that you wish to record*)*
       Tticker \[And] ticker == 1,
   Do[\[Rho]sol[Tticker][j] = \[Rho]array[[j + 1]], {j, 0, NR + 1}];
   Do[rsol[Tticker][j] = rarray[[j + 1]], {j, 0, NR + 1}];
   Do[\[Sigma]sol[Tticker][j] = \[Sigma]array[[j + 1]], {j, 0, 
     NR + 1}];
   T[Tticker] = Log10[t];
   Kn = Sqrt[
     4 \[Pi] G \[Rho]array[[1]]*\[Rho]0]*1/(\[Rho]array[[
       1]]*(\[Xi]array[[1]]/0.01*4578.17)*\[Sigma]array[[1]])*1/((
     r0 \[Rho]0)/t0);
   Print["Solved for t=" <> ToString[T[Tticker]] <> ", Kn=" <> 
     ToString[Kn]];
   Tticker = Tticker + 1;
   ticker = 0;,
   ticker = 1;
   ];
  If[(*\[Rho]array[[1]]>101.*)
   Log10[t] > 
    Log10[EVOLUTIONTIME],(*condition to finish computation*)
   Do[\[Rho]sol[Tticker][j] = \[Rho]array[[j + 1]], {j, 0, NR + 1}];
   Do[rsol[Ttiscker][j] = rarray[[j + 1]], {j, 0, NR + 1}];
   Do[\[Sigma]sol[Tticker][j] = \[Sigma]array[[j + 1]], {j, 0, 
     NR + 1}];
   T[Tticker] = Log10[t];
   Kn = Sqrt[
     4 \[Pi] G \[Rho]array[[1]] \[Rho]0]*1/(\[Rho]array[[
       1]]*(\[Xi]array[[1]]/0.01*4578.17)* \[Sigma]array[[1]])*1/((
     r0 \[Rho]0)/t0);
   Print["Solved for t=" <> ToString[T[Tticker]] <> ", Kn=" <> 
     ToString[Kn]];
   Break[];
   ];
  Clear[\[Delta]T];
  , {i, 0, 10^10}];
 Print["Exporting data..."];
 data\[Rho]solfinal = 
  Flatten[Table[{T[i], rsol[i][j], \[Rho]sol[i][j]}, {i, 0, 
     Tticker}, {j, 0, NR + 1}], 1];
 Export["\[Rho]sol_M_" <> ToString[GALAXYMASS] <> "_t_" <> 
    ToString[EVOLUTIONTIME] <> "_sigma_" <> ToString[CROSSSECTION] <> 
    ".txt", data\[Rho]solfinal, "Table"];
 data\[Sigma]solfinal = 
  Flatten[Table[{T[i], rsol[i][j], \[Sigma]sol[i][j]}, {i, 0, 
     Tticker}, {j, 0, NR + 1}], 1];
 Export["\[Sigma]sol_M_" <> ToString[GALAXYMASS] <> "_t_" <> 
    ToString[EVOLUTIONTIME] <> "_sigma_" <> ToString[CROSSSECTION] <> 
    ".txt", data\[Sigma]solfinal, "Table"];
 Print["Data export complete!"];
 , {p, 0, 0}];(*change it to {p,0,NM}, to scan halo masses*)
Print["Job ended."]

Exit[]

