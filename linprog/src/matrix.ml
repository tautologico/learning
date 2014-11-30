
type t = { 
  rows: int;
  cols: int;
  mxs: float array;
}

let create ?(initval=0.0) ~rows ~cols = 
  { rows; cols; mxs = Array.make (rows * cols) initval } 

let zero ~rows ~cols = create rows cols

let init ~f ~rows ~cols = 
  { rows; cols; mxs = Array.init (rows * cols) (fun i -> f (i mod cols) (i / cols)) }

let get m row col = m.mxs.(row * m.cols + col)

let set m row col v = m.mxs.(row * m.cols + col) <- v

let read ch ~rows ~cols = 
  init ~rows ~cols ~f:(fun _ _ -> Util.read_float ch)

let copy m = 
  { rows = m.rows; cols = m.cols; mxs = Array.copy m.mxs }

let transform_row m ~row ~f = 
  for col = 0 to (m.cols-1) do
    set m row col (f @@ get m row col)
  done

let transform_column m ~col ~f = 
  for r = 0 to (m.rows-1) do 
    set m r col (f @@ get m r col)
  done
