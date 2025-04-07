// Planar surface with an incident tine wire example
// Define a characteristic mesh size
lc = 0.2;

// Define the rectangle for the planar surface
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};



Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Define the tine wire as a line incident to the surface.
// The wire attaches at the midpoint of the top edge of the rectangle.
Point(5) = {0.5, 1, 0, lc};      // Common point on the surface
Point(6) = {0.5, 1.5, 0, lc};      // End point of the tine wire

Line(5) = {5, 6};

// Define physical groups for clarity and post-processing
Physical Surface("PlanarSurface") = {1};
Physical Curve("Boundary") = {1, 2, 3, 4};
Physical Curve("TineWire") = {5};
