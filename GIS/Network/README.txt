README

10-30-2023, 0-7123 Network Files Data Field Descriptions



*************************************************************************************************
ID - node name
LON_X - node longitude/x-coordinate
LAT_Y - node latitude/y-coordinate
COUNTY - name of county, if node is a county centroid
CNTY01 - 1 if node is county centroid, 0 otherwise

*************************************************************************************************


*************************************************************************************************
LINK_ID	- unique link identifier
LINK_TYPE - 	4 if centroid connector, 
		3 if toll road,
		2 if evaculane,
		1 if contraflow,
		0 if regular evacuation route
LENGTH_FT - link length in feet	
LENGTH_MI - link length in miles
LANES - total number of lanes in both directions for a typical cross section
SPD_LMT	- posted speed limit
CONTRAFLOW - 1 if link is designated for contraflow, 0 otherwise
EVACU_LANE - 1 if link is designated for evaculane, 0 otherwise
TOLL - 1 if link is a toll road, 0 otherwise
CAPACITY_A - link capacity from node A to node B, base condition
CAPACITY_B - link capacity from node B to node A, base condition
CAPACITY_1 - link capacity from A to B, including contraflow and evaculane treatments	
CAPACITY_2 - link capacity from B to A, including contraflow and evaculane treatments	
K_J_AB_BAS - jam density, base scenario, A to B
K_J_BA_BAS - jam density, base scenario, B to A
K_J_AB_ADD - jam density, additional capacity scenario, A to B
K_J_BA_ADD - jam density, additional capacity scenario, B to A
K_J_AB_TOL - jam density, toll lanes, A to B	
K_J_BA_TOL - jam density, toll lanes, B to A	
CAPACITY_3 - link capacity from A to B, for toll links
CAPACITY_4 - link capacity from B to A, for toll links
SPEED_TOLL - speed limit of toll links
TOLL_LANES - total number of lanes in both directions for a typical toll cross section
FROM_ID	- initial node
TO_ID - terminal node
FORMULA_FI - concatenation string of inital and terminal node

*************************************************************************************************
