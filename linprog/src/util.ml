
let read_int ic = 
  Scanf.fscanf ic " %d " (fun i -> i)

let read_float ic = 
  Scanf.fscanf ic " %f " (fun f -> f)

let read_vector ic ~size = 
  Array.init size (fun _ -> read_float ic)

let read_int_array ic ~size = 
  Array.init size (fun _ -> read_int ic)
