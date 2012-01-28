(* 
 *
 * id3.ml 
 * Simple ID3 implementation in OCaml
 * 
 * Andrei de A. Formiga, 2012-01-27
 * 
 *)

(* 
   - Test data is composed of objects. 
   - Each object is characterized by a set of attributes
   - Each attribute can assume a number of discrete values
*)

(* Attributes *)
type attribute = { name: string; nvals: int; repr: int -> string; reader: string -> int; }

(* An object is an array of attribute values *)
type obj = int array

(* The two object classes *)
type obj_class = Pos | Neg

(* Data representation *)
type dataset = 
    { dsname: string; 
      nattrs: int; 
      attrs: attribute array;
      objects: obj array; 
      classes: obj_class array; }


type dtree = Internal of attribute * dtree list | Leaf of obj_class


let gen_attr_set n = 
  let rec loop i = if i = n then [] else i :: loop (i+1) in 
  loop 0

let count_classes dset = 
  let count_case cls (p, n) = match cls with Pos -> (p+1, n) | Neg -> (p, n+1) in
  Array.fold_right count_case dset.classes (0,0)

let count_classes_attr dset attr_num = 
  let count_case obj cls counts = 
    let (p, n) = counts.(obj.(attr_num)) in
    match cls with 
        Pos -> counts.(obj.(attr_num)) <- (p+1, n)
      | Neg -> counts.(obj.(attr_num)) <- (p, n+1) in
  let counts = Array.make dset.attrs.(attr_num).nvals (0, 0) in
  Array.iteri (fun i o -> count_case o dset.classes.(i) counts) dset.objects;
  counts

let evaluate_entropy dset attr_num = 
  let attr = dset.attrs.(attr_num) in
  
let id3 dset = 
  let pos, neg = count_classes dset in
  let attr_set = gen_attr_set dset.nattrs in
  Leaf of Pos
