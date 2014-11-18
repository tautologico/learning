

let main () = 
  if Array.length Sys.argv < 2 then
    Printf.printf "Must specify the dictionary file on input\n"
  else
    let f = open_in Sys.argv.(1) in 
    let d = Dict.read f in
    Printf.printf "Pivoted!\n"
