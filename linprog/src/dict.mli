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
