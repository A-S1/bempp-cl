// ------------------------------------------------------------
// 1) Vertical wire at (0,0), from z=0 to z=1, split into 2 elements
// ------------------------------------------------------------
Point(1) = {0, 0, 0, 0.5};
Point(2) = {0, 0, 0.5, 0.5};
Point(3) = {0, 0, 1, 0.5};

Line(1) = {1, 2};
Line(2) = {2, 3};

Physical Line("Wire1") = {1, 2};

// ------------------------------------------------------------
// 2) L-shaped wire, offset by 1 in x, 
//    vertical branch (2 elements) pointing downward (–z),
//    horizontal branch (2 elements) along +x
// ------------------------------------------------------------
Point(4) = {1, 0, 1,   0.5};   // top of L
Point(5) = {1, 0, 0.5, 0.5};   // midway down
Point(6) = {1, 0, 0,   0.5};   // elbow of L
Point(7) = {1.5, 0, 0, 0.5};   // first horizontal split
Point(8) = {2, 0, 0,   0.5};   // end of L

Line(3) = {4, 5};
Line(4) = {5, 6};
Line(5) = {6, 7};
Line(6) = {7, 8};

Physical Line("Wire2") = {3, 4, 5, 6};

// ------------------------------------------------------------
// Mesh this as 1D
// ------------------------------------------------------------
Mesh 1;