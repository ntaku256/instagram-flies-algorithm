

    best = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    insta = InstaGramFlies(n_iters, n_clusters, n_indivisuals,r,master_w,master_c1,master_c2,master_c3,faddist_w,faddist_c1,faddist_c2,faddist_c3)
                    g_score,score, result = insta.Run()
                    best.append(g_score)
                    print(score,result)
                
                    faddist_c3 += 0.2
                faddist_c2 += 0.2
            faddist_c1 += 0.2
        faddist_w += 0.2


    write_wb = openpyxl.load_workbook("Books/Book_write.xlsx")
    write_ws = write_wb