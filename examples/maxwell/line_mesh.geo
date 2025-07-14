

// Define the points for the first line (vertical line at x = 0, y = 0)
Point(1) = {0, 0, 0, 1.0};  // Start point (x=0, y=0, z=0)
// The last parameter "1.0" is a mesh size at the point (can be adjusted as needed)
Point(2) = {0, 0, 0.2, 1.0};  // End point (x=0, y=0, z=1)

// Define the points for the second line (vertical line at x = 1, y = 0)
Point(3) = {0.2, 0, 0, 1.0};  // Start point (x=1, y=0, z=0)
Point(4) = {0.2, 0, 0.2, 1.0};  // End point (x=1, y=0, z=1)

// Create the two lines connecting the points
Line(1) = {1, 2};  // First vertical line
Line(2) = {3, 4};  // Second vertical line

// Subdivide each line into two elements.
// The transfinite command here sets 3 nodes per line (thus creating 2 segments).
Transfinite Line {1, 2} = 3;
