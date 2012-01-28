(* 
 *
 * id3.ml 
 * Simple ID3 implementation in OCaml
 * 
 * Andrei de A. Formiga, 2012-01-27
 * 
 *)


(* 
   TODO

   - Deal with an empty set of items for a certain attribute value, or 
   an empty set of attributes to partition the tree. 
*)



(* 
   - Test data is composed of objects. 
   - Each object is characterized by a set of attributes
   - Each attribute can assume a number of discrete values
*)

(* Attributes *)
type attribute = 
    { name: string; 
      nvals: int; 
      repr: int -> string; 
      reader: string -> int; }

(* An object is an array of attribute values *)
type obj = int array

(* The two object classes *)
type obj_class = Pos | Neg

type item = { obj: obj; cls: obj_class; }

(* Data representation *)
type dataset = 
    { dsname: string; 
      nattrs: int; 
      attrs: attribute array;
      items: item list; }

type dtree = Internal of attribute * dtree list | Leaf of obj_class

(* Base-2 logarithm *)
let log2 n = (log10 n) /. (log10 2.0)


(* Generate the initial attribute set (range 0..(n-1)) *)
let gen_attr_set n = 
  let rec loop i = if i = n then [] else i :: loop (i+1) in 
  loop 0

(* Count class membership numbers for the whole dataset *)
let count_classes dset = 
  let count_case item (p, n) = match item.cls with Pos -> (p+1, n) | Neg -> (p, n+1) in
  List.fold_right count_case dset.items (0,0)

(* Count class membership for all attribute values of attr_num *)
let count_classes_attr dset items attr_num = 
  let counts = Array.make dset.attrs.(attr_num).nvals (0, 0) in
  let count_case item = 
    let index = item.obj.(attr_num) in
    let (p, n) = counts.(index) in
    match item.cls with 
        Pos -> counts.(index) <- (p+1, n)
      | Neg -> counts.(index) <- (p, n+1) in
  List.iter count_case items;
  counts

(* calculates I(p, n) (expected information) for p positive and n negative objects *)
let exp_information p n = 
  let pf, nf = float p, float n in 
  let posfrac = pf /. (pf +. nf) in
  let negfrac = nf /. (pf +. nf) in
  if p = 0 || n = 0 then 0.0
  else -. (posfrac *. log2 posfrac) -. (negfrac *. log2 negfrac)

(* Evaluates the expected information gain for an attribute *)
let attr_information dset items attr_num = 
  let p, n = count_classes dset in
  let counts = count_classes_attr dset items attr_num in
  let calc_entropy pi ni = 
    let pif, nif = float pi, float ni in
    let pf, nf = float p, float n in 
    ((pif +. nif) /. (pf +. nf)) *. (exp_information pi ni) in
  Array.fold_right (fun (pi, ni) s -> s +. calc_entropy pi ni) counts 0.0 
  

(* 
   If attribute has many possible values, will call List.filter many times 
   on list of items. Possibly inefficient. 
*)
let partition_by_attribute dset items attr_num = 
  let attr = dset.attrs.(attr_num) in
  let partition_val i = List.filter (fun item -> item.obj.(attr_num) = i) items in
  Array.init attr.nvals partition_val

(* Remove item from list *)
let remove_attr attr_set attr = 
  List.fold_right (fun a lst -> if a = attr then lst else a :: lst) attr_set []
  
(* Check to see if all items are positive, negative or neither *)
let all_items_pos_neg items = 
  let rec loop items res = 
    match items with 
        [] -> Some res
      | i :: rest -> if i.cls = res then loop rest res else None in
  match items with 
      [] -> None
    | i :: rest -> loop rest i.cls

(* Induce a Decision Tree using the ID3 algorithm *)
let id3 dset = 
  let pos, neg = count_classes dset in
  let info = exp_information pos neg in
  let attr_set = gen_attr_set dset.nattrs in
  let compare_attr dset items attr (best, maxgain) = 
    let gain = info -. (attr_information dset items attr) in 
    if gain > maxgain then (attr, gain) else (best, maxgain) in
  let select_attr attr_set items = 
    fst (List.fold_right (compare_attr dset items) attr_set (0, 0.0)) in
  let rec partition dset attr_set items = 
    match all_items_pos_neg items with
        Some Pos -> Leaf Pos 
      | Some Neg -> Leaf Neg
      | None -> 
        let root_attr = select_attr attr_set items in
        let subtrees = Array.to_list (partition_by_attribute dset items root_attr) in
        let new_attr_set = remove_attr attr_set root_attr in
        Internal (dset.attrs.(root_attr), List.map (partition dset attr_set) subtrees) in
  partition dset attr_set dset.items


(* a test dataset *)
let weather_data = 
  { dsname = "Weather data";
    nattrs = 4;
    attrs = 
      [| { name = "Outlook";
           nvals = 3;   (* sunny = 0, overcast = 1, rain = 2 *)
           repr = (fun i -> "");
           reader = (fun s -> 0); }; 

         { name = "Temperature";
           nvals = 3;   (* cool = 0, mild = 1, hot = 2 *)
           repr = (fun i -> "");
           reader = (fun s -> 0); };

         { name = "Humidity";
           nvals = 2;   (* normal = 0, high = 1 *)
           repr = (fun i -> "");
           reader = (fun s -> 0); };

         { name = "Windy";
           nvals = 2;   (* false = 0, true = 1 *)
           repr = (fun i -> "");
           reader = (fun s -> 0); };
      |];

    items = 
      [
        { obj = [| 0; 2; 1; 0 |]; cls = Neg; };
        { obj = [| 0; 2; 1; 1 |]; cls = Neg; };
        { obj = [| 1; 2; 1; 0 |]; cls = Pos; };
        { obj = [| 2; 1; 1; 0 |]; cls = Pos; };
        { obj = [| 2; 0; 0; 0 |]; cls = Pos; };
        { obj = [| 2; 0; 0; 1 |]; cls = Neg; };
        { obj = [| 1; 0; 0; 1 |]; cls = Pos; };
        { obj = [| 0; 1; 1; 0 |]; cls = Neg; };
        { obj = [| 0; 0; 0; 0 |]; cls = Pos; };
        { obj = [| 2; 1; 0; 0 |]; cls = Pos; };
        { obj = [| 0; 1; 0; 1 |]; cls = Pos; };
        { obj = [| 1; 1; 1; 1 |]; cls = Pos; };
        { obj = [| 1; 2; 0; 0 |]; cls = Pos; };
        { obj = [| 2; 1; 1; 1 |]; cls = Neg; }
      ];      
  }

(* Testing with the example from Quinlan's paper "Induction of Decision Trees" *)
let test = id3 weather_data
