// Simple planar surface example
// Define a characteristic mesh size
lc = 0.1;

// Define the four corner points of a rectangle
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Define the four lines connecting the points
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Create a closed line loop from the lines
Line Loop(1) = {1, 2, 3, 4};

// Define the planar surface bounded by the line loop
Plane Surface(1) = {1};

// Define physical groups for post-processing (optional)
Physical Surface("PlanarSurface") = {1};



