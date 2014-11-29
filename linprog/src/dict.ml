
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

type pivot_step_ix = { 
  enter_ix : int;
  leave_ix : int;
  objvalue: float
}

type pivot_result = FinalDict | UnboundedDict | NormalStep of pivot_step

type solve_result = Unbounded | SolutionFound of int * t

type solution = {
  steps: int;
  final_val: float
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

let read_file = Util.read_val_from_file read

let analyze_entering d = 
  Array.mapi (fun i x -> (x,i-1)) d.obj
  |> Array.to_list
  |> List.filter (fun (x,i) -> x > 0.0 && i >= 0)
  |> List.map (fun (_,i) -> d.nbasic.(i), i)
  |> Util.lst_foldl1 (fun (x1,i1) (x2,i2) -> if x2 < x1 then (x2,i2) else (x1,i1))
  |> Util.lift_option snd 

let find_min_pos_index dict v = 
  let select_pair (i1,x1) (i2,x2) = 
    let v1, v2 = dict.basic.(i1), dict.basic.(i2) in
    if x1 < x2 then (i1,x1) 
    else if x1 <= x2 && v1 < v2 then (i1,x1)
    else (i2,x2)
  in
  List.mapi (fun i x -> (i, x)) v 
  |> List.filter (fun (_,x) -> x >= 0.0)
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

let read_pivot_step_file = Util.read_val_from_file read_pivot_step 
 
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
                          { entering = dict.nbasic.(enter); leaving = dict.basic.(v);
                            objval = c *. dict.obj.(enter+1) +. dict.obj.(0) })

let calc_pivot_step_ix dict = 
  match analyze_entering dict with
  | None -> None
  | Some enter -> 
     analyze_leaving dict enter 
     |> Util.lift_option (fun (v, c) -> 
                          { enter_ix = enter; leave_ix = v;
                            objvalue = c *. dict.obj.(enter+1) +. dict.obj.(0) })

let eq_pivot_step ps1 ps2 = 
  ps1.entering = ps2.entering 
  && ps1.leaving = ps2.leaving
  && Util.float_eq ps1.objval ps2.objval 0.0001

let read_solution ic = 
  let first = Scanf.fscanf ic " %s " (fun s -> s) in
  if first = "UNBOUNDED" then None
  else 
    let final_val = Scanf.sscanf first " %f " (fun f -> f) in
    Scanf.fscanf ic " %d " 
                 (fun steps -> 
                  Some { steps; final_val })

let read_solution_file = Util.read_val_from_file read_solution

let eq_solution s1 s2 = 
  s1.steps = s2.steps 
  && Util.float_eq s1.final_val s2.final_val 0.0001

let write_dict oc dict = 
  let write_basis_line i x = 
    Printf.fprintf oc "x%u | %05.4f " x dict.assign.(i);
    Array.iteri (fun j y -> 
                 Printf.fprintf oc "%05.4fx%u " (Matrix.get dict.a i j) y) 
                dict.nbasic;
    Printf.fprintf oc "\n"
  in
  Array.iteri write_basis_line dict.basic;
  Printf.fprintf oc "--------------------------------\n";
  Printf.fprintf oc "z  | %05.4f " dict.obj.(0);
  Array.iteri (fun i x -> Printf.fprintf oc "%05.4fx%u " dict.obj.(i+1) x) dict.nbasic;
  Printf.fprintf oc "\n"

let print_dict dict = 
  write_dict stdout dict 

let pivot dict psi = 
  let n_basic = Array.copy dict.basic in
  let n_nbasic = Array.copy dict.nbasic in
  n_basic.(psi.leave_ix) <- n_nbasic.(psi.enter_ix);
  n_nbasic.(psi.enter_ix) <- n_basic.(psi.leave_ix);

  (* invert equation for leaving variable *)
  let n_a = Matrix.copy dict.a in
  let l_factor = -. (Matrix.get n_a psi.leave_ix psi.enter_ix) in
  Matrix.set n_a psi.leave_ix psi.enter_ix (-. 1.0);
  Matrix.transform_column n_a psi.leave_ix (fun x -> x /. l_factor);
  let n_assign = Array.copy dict.assign in
  n_assign.(psi.leave_ix) <- n_assign.(psi.leave_ix) /. l_factor; 

  (* adjust other equations *)
  for r = 0 to (psi.leave_ix-1) do  (* TODO: rewrite to avoid repetition *)
    let x1 = Matrix.get n_a r psi.enter_ix in
    for c = 0 to (dict.n-1) do
      let x2 = Matrix.get n_a psi.leave_ix c in
      Matrix.set n_a r c ((Matrix.get dict.a r c) +. x1 *. x2)
    done;
    let x3 = Matrix.get n_a psi.leave_ix psi.enter_ix in
    Matrix.set n_a r psi.enter_ix (x1 *. x3);
    n_assign.(r) <- n_assign.(r) +. x1 *. n_assign.(psi.leave_ix)
  done;

  for r = (psi.leave_ix+1) to (dict.m-1) do
    let x1 = Matrix.get dict.a r psi.enter_ix in
    for c = 0 to (dict.n-1) do 
      let x2 = Matrix.get n_a psi.leave_ix c in
      Matrix.set n_a r c ((Matrix.get dict.a r c) +. x1 *. x2)
    done;
    let x3 = Matrix.get n_a psi.leave_ix psi.enter_ix in
    Matrix.set n_a r psi.enter_ix (x1 *. x3);
    n_assign.(r) <- n_assign.(r) +. x1 *. n_assign.(psi.leave_ix)
  done;

  (* adjust object function values *)
  let n_obj = Array.copy dict.obj in
  n_obj.(0) <- psi.objvalue; 
  for i = 1 to dict.n do
    n_obj.(i) <- dict.obj.(i) +. dict.obj.(psi.enter_ix+1) *. (Matrix.get n_a psi.leave_ix (i-1))
  done;
  n_obj.(psi.enter_ix+1) <- dict.obj.(psi.enter_ix+1) *. (Matrix.get n_a psi.leave_ix psi.enter_ix);

  (* build new dict *)
  { m = dict.m; n = dict.n; basic = n_basic; nbasic = n_nbasic;
    assign = n_assign; a = n_a; obj = n_obj }


