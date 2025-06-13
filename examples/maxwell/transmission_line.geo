// Transmission line boundary loops for Gmsh (no surfaces)

// Parameters
r   = 0.01;    // Conductor radius [m]
d   = 0.2;     // Separation between conductor centers [m]
lc  = 0.005;   // Characteristic mesh size
w   = 1.0;     // Width of surrounding domain [m]
h   = 0.5;     // Height of surrounding domain [m]

// Conductor 1 (center at +d/2,0)
x1 =  d/2; y1 =  0;
Point(1)  = {x1 + r, y1,    0, lc};
Point(2)  = {x1,      y1 + r,0, lc};
Point(3)  = {x1 - r,  y1,    0, lc};
Point(4)  = {x1,      y1 - r,0, lc};
Circle(5) = {2, 1, 4};
Circle(6) = {3, 2, 1};
Circle(7) = {4, 3, 2};
Circle(8) = {1, 4, 3};
Curve Loop(101) = {5,6,7,8};

// Conductor 2 (center at -d/2,0)
x2 = -d/2; y2 =  0;
Point(11) = {x2 + r, y2,    0, lc};
Point(12) = {x2,      y2 + r,0, lc};
Point(13) = {x2 - r,  y2,    0, lc};
Point(14) = {x2,      y2 - r,0, lc};
Circle(15) = {12,11,14};
Circle(16) = {13,12,11};
Circle(17) = {14,13,12};
Circle(18) = {11,14,13};
Curve Loop(102) = {15,16,17,18};

// Outer boundary loop (rectangle)
Point(21) = {-w/2, -h/2, 0, lc};
Point(22) = { w/2, -h/2, 0, lc};
Point(23) = { w/2,  h/2, 0, lc};
Point(24) = {-w/2,  h/2, 0, lc};
Line(25)  = {21,22};
Line(26)  = {22,23};
Line(27)  = {23,24};
Line(28)  = {24,21};
Curve Loop(103) = {25,26,27,28};

// Physical groups of line loops (optional)
Physical Curve("Cond1Loop")   = {5,6,7,8};
Physical Curve("Cond2Loop")   = {15,16,17,18};
Physical Curve("OuterLoop")   = {25,26,27,28};
