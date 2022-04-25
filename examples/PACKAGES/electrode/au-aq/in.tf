include "in.settings"

fix conp bot electrode/conp -1.0 1.805132 couple top 1.0 symm on etypes 6*7
fix_modify conp tf 6 1.0 18.1715745
fix_modify conp tf 7 1.0 18.1715745
thermo_style custom step temp c_ctemp epair etotal c_qtop c_qbot
run 500
