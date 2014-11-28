
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

let lift_eq_option cmp o1 o2 = 
  match o1, o2 with 
  | None, None -> true
  | Some x1, Some x2 -> cmp x1 x2 
  | _ -> false

let float_eq f1 f2 tol = 
  abs_float (f1 -. f2) <= tol 

let range_lst n1 n2 = 
  let rec loop i = 
    if i > n2 then [] else i :: loop (i+1) in
  loop n1 

let read_val_from_file read_fun fname = 
  let f = open_in fname in
  let v = read_fun f in
  close_in f;
  v
