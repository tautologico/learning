
(** Type for matrices *)
type t

(** Creates a matrix of given size, filled with initial 
    value [initval] (default is 0.0). *)
val create : ?initval:float -> rows:int -> cols:int -> t

(** Create a zero matrix of given size. *)
val zero : rows:int -> cols:int -> t

(** [get m i j] gets the value of the element at row [i] and 
    column [j] of matrix [m]. *)
val get : t -> int -> int -> float

(** [set m i j v] sets the value of the element at row [i] and 
    column [j] of matrix [m]. *)
val set : t -> int -> int -> float -> unit

(** Reads a matrix from an input channel. *)
val read : in_channel -> rows:int -> cols:int -> t 

(** Copy a matrix *)
val copy : t -> t 

val transform_row : t -> row:int -> f:(float -> float) -> unit 

val transform_column : t -> col:int -> f:(float -> float) -> unit
