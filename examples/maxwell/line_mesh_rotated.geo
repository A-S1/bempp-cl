

// mesh size
h = 1.0;
// rotation angle
angle = Pi/4;
c = Cos(angle);  s = Sin(angle);

// first (unchanged) vertical line at the origin
Point(1) = {0,   0, 0, h};
Point(2) = {0,   0, 1, h};

// second line, rotated
Point(3) = {c,   s, 0, h};
Point(4) = {c,   s, 1, h};

Line(1) = {1, 2};
Line(2) = {3, 4};

// split each line into 2 segments (3 nodes per line)
Transfinite Line {1, 2} = 3;

