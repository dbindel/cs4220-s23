function silly_sqrt()
    x = 2.0
    for k = 1:60  x = sqrt(x)  end
    for k = 1:60  x = x^2      end
    println(x)
end
