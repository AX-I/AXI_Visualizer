# Extended .obj file format for bone weights

import numpy as np

def exportObjMultiTexNormBones(fn, mtln, texs, tris, uvs, nms, bws, mtlAlias=None):
    with open(fn, "w") as f:
        f.write("# Extended OBJ File exported from AXI Creator\n")
        f.write("# Includes vertex bone assignment (nearest)\n")
        points = {}
        tcoords = {}
        norms = {}
        L = 1
        M = 1
        N = 1
        texlens = []
        indices = []
        uindices = []
        nindices = []
        bindices = []
        for tn in range(len(texs)):
            tri = tris[tn]
            uv = uvs[tn]
            norm = nms[tn]
            bw = bws[tn]
            texlens.append(len(tri))
            for i in range(len(tri)):
                t = tri[i]
                vt = uv[i]
                nt = norm[i]
                ix = []
                ux = []
                nx = []
                for j in range(3):
                    try:
                        ix.append(points[tuple(t[j])])
                    except KeyError:
                        points[tuple(t[j])] = L
                        ix.append(L)
                        L += 1
                    try:
                        ux.append(tcoords[tuple(vt[j])])
                    except KeyError:
                        tcoords[tuple(vt[j])] = M
                        ux.append(M)
                        M += 1
                    try:
                        nx.append(norms[tuple(nt[j])])
                    except KeyError:
                        norms[tuple(nt[j])] = N
                        nx.append(N)
                        N += 1
                indices.append(ix)
                uindices.append(ux)
                nindices.append(nx)
                bindices.append(bw[i])
        
        f.write("# " + str(L-1) + " vertices\n")
        f.write("# " + str(M-1) + " texcoords\n")
        f.write("# " + str(N-1) + " normals\n")
        f.write("# " + str(max([np.max(b) for b in bws])) + \
                " bones excluding root\n")
        f.write("# " + str(sum(texlens)) + " triangles\n\n")

        if mtlAlias is None:
            f.write("mtllib " + mtln + "\n")
        else:
            f.write("mtllib " + mtlAlias + "\n")

        for p in points:
            f.write("v " + " ".join([str(round(x,8)) for x in [p[0], p[1], -p[2]]]) + "\n")
        for n in norms:
            f.write("vn " + " ".join([str(round(-x,8)) for x in [n[0], n[1], -n[2]]]) + "\n")
        for u in tcoords:
            f.write("vt " + str(round(u[1],8)) + " " + str(round(u[0],8)) + "\n")

        tstart = 0
        for tn in range(len(texlens)):
            f.write("usemtl Mat" + str(tn) + "\n")
            tlen = texlens[tn]
            for i in range(tstart, tstart + tlen):
                f.write("f ")
                ix = indices[i]
                ux = uindices[i]
                nx = nindices[i]
                bx = bindices[i]
                for n in range(3):
                    f.write(str(ix[n]) + "/" + str(ux[n]) + "/")
                    f.write(str(nx[n]) + "/" + str(bx[n]) + " ")
                f.write("\n")
            tstart += tlen
        
        f.write("# End of file\n")
    
    with open(mtln, "w") as f:
        f.write("# MTL file exported from AXI Creator\n")
        for tn in range(len(texs)):
            f.write("newmtl Mat" + str(tn) + "\n")
            f.write("map_Kd " + texs[tn] + "\n")
