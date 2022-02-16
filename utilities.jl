using PyCall

pickle = pyimport("pickle")

function unpickler(filename)

  r = nothing
  @pywith pybuiltin("open")(filename,"rb") as f begin
    r = pickle.load(f)
  end
  return r

end

function pickler(filename, obj)

  out = open(filename,"w")
  pickle.dump(obj, out)
  close(out)

 end
