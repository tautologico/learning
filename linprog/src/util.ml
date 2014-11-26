
let read_int ic = 
  Scanf.fscanf ic " %d " (fun i -> i)

let read_float ic = 
  Scanf.fscanf ic " %f " (fun f -> f)

let read_vector ic ~size = 
  Array.init size (fun _ -> read_float ic)

let read_int_array ic ~size = 
  Array.init size (fun _ -> read_int ic)

let lst_foldl1 f l = 
  match l with
    [] -> None
  | x :: rl -> Some (List.fold_left f x rl)

let lift_option f opt = 
  match opt with
  | None -> None
  | Some x -> Some (f x) 
