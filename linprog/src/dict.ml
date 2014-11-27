
type t = {
  m: int;
  n: int;
  basic: int array;
  nbasic: int array;
  assign: float array;
  a : Matrix.t;
  obj : float array;
}

type pivot_step = { 
  entering : int;
  leaving : int;
  objval : float
}

let read ch = 
  let m = Util.read_int ch in
  let n = Util.read_int ch in
  let basic = Util.read_int_array ch m in
  let nbasic = Util.read_int_array ch n in
  let assign = Util.read_vector ch m in
  let a = Matrix.read ch ~rows:m ~cols:n in
  let obj = Util.read_vector ch (n+1) in
  { m; n; basic; nbasic; assign; a; obj }

let read_file fname = 
  let f = open_in fname in
  let d = read f in
  close_in f;
  d

let analyze_entering d = 
  Array.mapi (fun i x -> (x,i-1)) d.obj
  |> Array.to_list
  |> List.filter (fun (x,i) -> x > 0.0 && i >= 0)
  |> List.map (fun (_,i) -> d.nbasic.(i), i)
  |> Util.lst_foldl1 (fun (x1,i1) (x2,i2) -> if x2 < x1 then (x2,i2) else (x1,i1))
  |> Util.lift_option snd 

let find_min_pos_index dict v = 
  let select_pair (i1,x1) (i2,x2) = 
    if x1 < x2 then (i1,x1) 
    else if x1 <= x2 && i1 < i2 then (i1,x1)
    else (i2,x2)
  in
  List.mapi (fun i x -> (i, x)) v 
  |> List.filter (fun (_,x) -> x >= 0.0)
  |> List.map (fun (i,x) -> (dict.basic.(i), x))
  |> Util.lst_foldl1 select_pair 

let analyze_leaving d enter = 
  Array.to_list d.assign 
  |> List.mapi (fun i x -> -. x /. (Matrix.get d.a i enter))
  |> find_min_pos_index d

let read_pivot_step ic = 
  let first = Scanf.fscanf ic " %s " (fun s -> s) in
  if first = "UNBOUNDED" then None
  else 
    let enter = Scanf.sscanf first "%d" (fun i -> i) in
    Scanf.fscanf ic " %d\n %f " 
                 (fun leave obj -> 
                  Some { entering = enter; leaving = leave; objval = obj })

let read_pivot_step_file fname = 
  let f = open_in fname in
  let ps = read_pivot_step f in
  close_in f;
  ps
    
let show_pivot_step_opt ops = 
  match ops with
  | None -> "UNBOUNDED"
  | Some ps -> Printf.sprintf "%d\n%d\n%5.3f\n" ps.entering ps.leaving ps.objval

let calc_pivot_step dict = 
  match analyze_entering dict with
  | None -> None
  | Some enter -> 
     analyze_leaving dict enter 
     |> Util.lift_option (fun (v, c) -> 
                          { entering = dict.nbasic.(enter); leaving = v;
                            objval = c *. dict.obj.(enter+1) +. dict.obj.(0) })

let eq_tolerance f1 f2 tol = 
  abs_float (f1 -. f2) <= tol 

let eq_pivot_step ps1 ps2 = 
  ps1.entering = ps2.entering 
  && ps1.leaving = ps2.leaving
  && eq_tolerance ps1.objval ps2.objval 0.0001


