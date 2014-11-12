
let read_int ic = 
  Scanf.fscanf ic " %d " (fun i -> i)

let read_float ic = 
  Scanf.fscanf ic " %f " (fun f -> f)

