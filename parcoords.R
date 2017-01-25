library(parcoords)
a <- format(date_seq, "%Y")
#parcoords(usedata)


parcoords(usedata, rownames = F,
          brushMode = "1D-axes-multi", 
          reorderable = TRUE)


