function testnan()
    x = 0.0/0.0;
    if x < 0.0       println("x is negative")
    elseif x >= 0.0  println("x is non-negative")
    else             println("Uh...")
    end
end
