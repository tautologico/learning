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

type pivot_result = FinalDict | UnboundedDict | NormalStep of pivot_step_ix

type solve_result = Unbounded | SolutionFound of int * t

type solution = {
  steps: int;
  final_val: float
}

val read : in_channel -> t 

val read_file : string -> t 

(* TODO: remove from interface *)
val find_min_pos_index : t -> float list -> (int * float) option

val analyze_entering : t -> int option 

val analyze_leaving : t -> int -> (int * float) option 

val read_pivot_step : in_channel -> pivot_step option

val read_pivot_step_file : string -> pivot_step option 

val show_pivot_step_opt : pivot_step option -> string 

val calc_pivot_step : t -> pivot_step option 

val eq_pivot_step : pivot_step -> pivot_step -> bool 

val write_dict : out_channel -> t -> unit

val print_dict : t -> unit

val read_solution : in_channel -> solution option

val read_solution_file : string -> solution option 

val eq_solution : solution -> solution -> bool 

val solve_lp : t -> bool -> solve_result
