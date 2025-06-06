// line.geo

// Mesh size parameter (not critical when using Transfinite)
lc = 1.0;

// Define the two end points
Point(1) = { 0, 0, -1, lc };
Point(2) = {  0, 0, 1, lc };

// Create the straight line between them
Line(1) = { 1, 2 };

// Force 21 nodes (→20 elements) equally spaced along Line 1
Transfinite Line { 1 } = 21 Using Progression 1;

// (Optional) Tag the line as a physical entity
Physical Line("Segment") = { 1 };

// Generate the 1D mesh
Mesh 1;
