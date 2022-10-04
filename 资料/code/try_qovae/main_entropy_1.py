import numpy
import copy

devices = ['BS(XXX,a,b)', 'BS(XXX,a,c)',
           'BS(XXX,a,d)', 'BS(XXX,a,e)',
           'BS(XXX,a,f)', 'BS(XXX,b,c)',
           'BS(XXX,b,d)', 'BS(XXX,b,e)',
           'BS(XXX,b,f)', 'BS(XXX,c,d)',
           'BS(XXX,c,e)', 'BS(XXX,c,f)',
           'BS(XXX,d,e)', 'BS(XXX,d,f)',
           'BS(XXX,e,f)',
           'DownConv(XXX,1,a,b)', 'DownConv(XXX,1,a,c)', 'DownConv(XXX,1,a,d)',
           'DownConv(XXX,1,a,e)', 'DownConv(XXX,1,a,f)', 'DownConv(XXX,1,b,c)',
           'DownConv(XXX,1,b,d)', 'DownConv(XXX,1,b,e)', 'DownConv(XXX,1,b,f)',
           'DownConv(XXX,1,c,d)', 'DownConv(XXX,1,c,e)', 'DownConv(XXX,1,c,f)',
           'DownConv(XXX,1,d,e)', 'DownConv(XXX,1,d,f)', 'DownConv(XXX,1,e,f)',
           'Reflection(XXX,a)', 'DP(XXX,a)',
           'OAMHolo(XXX,a,1)', 'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,a,3)', 'OAMHolo(XXX,a,4)', 'OAMHolo(XXX,a,5)',
           'OAMHolo(XXX,a,-1)', 'OAMHolo(XXX,a,-2)', 'OAMHolo(XXX,a,-3)', 'OAMHolo(XXX,a,-4)', 'OAMHolo(XXX,a,-5)',
           'Reflection(XXX,b)', 'DP(XXX,b)',
           'OAMHolo(XXX,b,1)', 'OAMHolo(XXX,b,2)', 'OAMHolo(XXX,b,3)', 'OAMHolo(XXX,b,4)', 'OAMHolo(XXX,b,5)',
           'OAMHolo(XXX,b,-1)', 'OAMHolo(XXX,b,-2)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,b,-4)', 'OAMHolo(XXX,b,-5)',
           'Reflection(XXX,c)', 'DP(XXX,c)',
           'OAMHolo(XXX,c,1)', 'OAMHolo(XXX,c,2)', 'OAMHolo(XXX,c,3)', 'OAMHolo(XXX,c,4)', 'OAMHolo(XXX,c,5)',
           'OAMHolo(XXX,c,-1)', 'OAMHolo(XXX,c,-2)', 'OAMHolo(XXX,c,-3)', 'OAMHolo(XXX,c,-4)', 'OAMHolo(XXX,c,-5)',
           'Reflection(XXX,d)', 'DP(XXX,d)',
           'OAMHolo(XXX,d,1)', 'OAMHolo(XXX,d,2)', 'OAMHolo(XXX,d,3)', 'OAMHolo(XXX,d,4)', 'OAMHolo(XXX,d,5)',
           'OAMHolo(XXX,d,-1)', 'OAMHolo(XXX,d,-2)', 'OAMHolo(XXX,d,-3)', 'OAMHolo(XXX,d,-4)', 'OAMHolo(XXX,d,-5)',
           'Reflection(XXX,e)', 'DP(XXX,e)',
           'OAMHolo(XXX,e,1)', 'OAMHolo(XXX,e,2)', 'OAMHolo(XXX,e,3)', 'OAMHolo(XXX,e,4)', 'OAMHolo(XXX,e,5)',
           'OAMHolo(XXX,e,-1)', 'OAMHolo(XXX,e,-2)', 'OAMHolo(XXX,e,-3)', 'OAMHolo(XXX,e,-4)', 'OAMHolo(XXX,e,-5)',
           'Reflection(XXX,f)', 'DP(XXX,f)',
           'OAMHolo(XXX,f,1)', 'OAMHolo(XXX,f,2)', 'OAMHolo(XXX,f,3)', 'OAMHolo(XXX,f,4)', 'OAMHolo(XXX,f,5)',
           'OAMHolo(XXX,f,-1)', 'OAMHolo(XXX,f,-2)', 'OAMHolo(XXX,f,-3)', 'OAMHolo(XXX,f,-4)', 'OAMHolo(XXX,f,-5)',
           ' ']

devices_cut = {'BS': 0, 'Do': 1, 'Re': 2, 'DP': 3, 'OA': 4}


def create_initial_state(ABCD):
    ABCD[0]['0 0'], ABCD[9]['0 0'] = 1, 1


def find_function(exp):
    _ = devices_cut[exp[0:2]]
    return _


def find_space(key):
    return key.find(' ')


def find_xp_x(num):
    _ = ['a', 'b', 'c', 'd', 'e', 'f']
    return _[num]


def find_p(p):  # eg: ab 0 0 1 bc 5 0 2
    P_R = {'a': [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
           'b': [[0, 5, 6, 7, 8], [1, 0, 0, 0, 0], [0, 2, 3, 4, 5]],
           'c': [[1, 5, 9, 10, 11], [1, 1, 0, 0, 0], [0, 1, 3, 4, 5]],
           'd': [[2, 6, 9, 12, 13], [1, 1, 1, 0, 0], [0, 1, 2, 4, 5]],
           'e': [[3, 7, 10, 12, 14], [0, 0, 0, 0, 1], [0, 1, 2, 3, 5]],
           'f': [[4, 8, 11, 13, 14], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]}
    return P_R[p]


def find_idx_p_pp(p, pp):
    IDX = {'ab': 0, 'ac': 1, 'ad': 2, 'ae': 3, 'af': 4, 'bc': 5, 'bd': 6, 'be': 7, 'bf': 8, 'cd': 9, 'ce': 10, 'cf': 11,
           'de': 12, 'df': 13, 'ef': 14}
    # p>pp
    return IDX[p + pp]


def change_p_pp(p, pp, ABCD, new_ABCD):
    idx_1 = find_p(p)
    for _ in range(5):
        set = ABCD[idx_1[0][_]]
        change_idx = idx_1[1][_]
        another_letter = find_xp_x(idx_1[2][_])

        new_set = new_ABCD[idx_1[0][_]]
        for key, value in set.items():
            idx_space = find_space(key)
            if change_idx == 0:
                new_key = str(-int(key[:idx_space])) + key[idx_space:]
            else:
                new_key = key[:idx_space + 1] + str(-int(key[idx_space + 1:]))
            new_set[new_key] = (1j * value) / numpy.sqrt(2)
            if pp != another_letter:
                if pp < another_letter:
                    another_letter_pp_idx = find_idx_p_pp(pp, another_letter)
                    pp_change = 0
                else:
                    another_letter_pp_idx = find_idx_p_pp(another_letter, pp)
                    pp_change = 1
                new_set_pp = new_ABCD[another_letter_pp_idx]
                if change_idx == pp_change:
                    new_set_pp[key] = value / numpy.sqrt(2)
                else:
                    new__key = key[idx_space + 1:] + ' ' + key[:idx_space]
                    new_set_pp[new__key] = value / numpy.sqrt(2)
        ABCD[idx_1[0][_]] = {}


def BS(exp, ABCD):  # "BS(XXX,a,d)"
    # print(ABCD)
    p, pp = exp[-4], exp[-2]
    new_ABCD = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    # 先处理p,pp项
    idx_p_pp = find_idx_p_pp(p, pp)
    phi = ABCD[idx_p_pp]
    for key, value in phi.items():
        # print(key,'@')
        idx_space = find_space(key)
        key_1 = key[idx_space + 1:] + ' ' + key[:idx_space]
        # print(key[:idx_space], key[idx_space + 1:])
        key_2 = str(-int(key[:idx_space])) + ' ' + str(-int(key[idx_space + 1:]))
        new_ABCD[idx_p_pp][key_1] = value / 2
        new_ABCD[idx_p_pp][key_2] = -value / 2
    ABCD[idx_p_pp] = {}

    # 从p开始
    # 找到p对应的位置
    change_p_pp(p, pp, ABCD, new_ABCD)
    change_p_pp(pp, p, ABCD, new_ABCD)

    for _ in range(15):
        phi = ABCD[_]
        new_phi = new_ABCD[_]
        for key, value in new_phi.items():
            inital = phi.get(key, 0)
            phi[key] = inital + new_phi[key]


def Do(exp, ABCD):  # DownConv(XXX,1,a,d)
    l_oam, p, pp = exp[-6], exp[-4], exp[-2]
    l_oam = int(l_oam)
    idx = find_idx_p_pp(p, pp)
    phi = ABCD[idx]
    for lorder in range(-l_oam, l_oam + 1):
        key = str(lorder) + ' ' + str(-lorder)
        inital = phi.get(key, 0)
        phi[key] = inital + 1


def Re(exp, ABCD):
    _ = exp[-2]
    P_R = find_p(_)
    for __ in range(5):
        i = P_R[0][__]
        ii = P_R[1][__]
        phi = ABCD[i]
        new_phi = {}
        for key, value in phi.items():
            phi[key] = 0
            idx = find_space(key)
            if ii == 0:
                new_key = str(-int(key[:idx])) + key[idx:]
            else:
                new_key = key[:idx + 1] + str(-int(key[idx + 1:]))
            initial = phi.get(new_key, 0)
            new_phi[new_key] = 1j * value + initial
        ABCD[i] = new_phi


def DP(exp, ABCD):  # 'DP(XXX,b)'
    _ = exp[-2]
    P_R = find_p(_)
    for __ in range(5):
        i = P_R[0][__]
        ii = P_R[1][__]
        phi = ABCD[i]
        new_phi = {}
        for key, value in phi.items():
            phi[key] = 0
            idx = find_space(key)
            if ii == 0:
                new_key = str(-int(key[:idx])) + key[idx:]
                lost = (-1) ** (abs(int(key[:idx])) // 2)
            else:
                new_key_p = str(-int(key[idx + 1:]))
                new_key = key[:idx + 1] + new_key_p
                lost = (-1) ** (abs(int(key[idx + 1:])) // 2)
            initial = phi.get(new_key, 0)
            new_phi[new_key] = 1j * lost * value + initial
        ABCD[i] = new_phi


def OA(exp, ABCD):
    _ = exp[-3]
    # get p,n
    if _ == '-':
        p = exp[-5]
        n = exp[-3] + exp[-2]
    else:
        p = exp[-4]
        n = exp[-2]
    P_R = find_p(p)
    for __ in range(5):  # 5 is the max_num of path
        i = P_R[0][__]
        ii = P_R[1][__]
        phi = ABCD[i]
        new_phi = {}
        for key, value in phi.items():
            phi[key] = 0
            idx = find_space(key)
            if ii == 0:
                new_key_p = int(key[:idx]) + int(n)
                new_key = str(new_key_p) + key[idx:]
            else:
                new_key_p = int(key[idx + 1:]) + int(n)
                new_key = key[:idx] + ' ' + str(new_key_p)
            initial = phi.get(new_key, 0)
            new_phi[new_key] = value + initial
        ABCD[i] = new_phi


# print(type(ABCD), "\n", ABCD)
example = ['DownConv(XXX,1,a,d)','DownConv(XXX,1,b,c)','BS(XXX,b,e)']
# example = ['DownConv(XXX,1,b,f)', 'OAMHolo(XXX,c,2)', 'DownConv(XXX,1,a,b)']
# example = ['OAMHolo(XXX,b,-1)','BS(XXX,a,b)','DP(XXX,f)']


def find_key_ijkl(key):  # 给key返回ijkl值
    idx_1 = key.find(' ')
    idx_2 = key.find(' ', idx_1 + 1)
    idx_3 = key.rfind(' ')
    ijkl = [key[:idx_1], key[idx_1 + 1:idx_2], key[idx_2 + 1:idx_3], key[idx_3 + 1:]]
    return ijkl


def comput_S_p_x(phi_4, x):  # abcd计算纠缠熵
    p_x = {}
    a = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    idx = a[x]
    # 建立p_x
    for key, value in phi_4.items():
        for key_1, value_1 in phi_4.items():
            ijkl = find_key_ijkl(key)
            ijkl_1 = find_key_ijkl(key_1)
            ijkl_x = copy.deepcopy(ijkl)
            ijkl_1_x = copy.deepcopy(ijkl_1)
            ijkl_x.pop(idx)
            ijkl_1_x.pop(idx)
            if ijkl_x == ijkl_1_x:
                new_key = ijkl[idx] + ' ' + ijkl_1[idx]
                inital = p_x.get(new_key, 0)
                p_x[new_key] = value * value_1.conjugate() + inital
    # 计算p_x
    idx_matrix = {}
    ii = 0
    for key, value in p_x.items():
        idx = find_space(key)
        if key[:idx] == key[idx + 1:]:
            idx_matrix[key[:idx]] = ii
            ii += 1
    matrix = numpy.zeros((ii, ii))
    for key_1, value_1 in p_x.items():
        idx_1 = find_space(key_1)
        aa = idx_matrix[key_1[:idx_1]]
        bb = idx_matrix[key_1[idx_1 + 1:]]
        matrix[aa][bb] = value_1.real
    w, v = numpy.linalg.eig(matrix)
    Sum = 0
    for ww in w:
        if ww > 1e-8:
            Sum += -ww * numpy.log(ww)
    return Sum


def comput_S_ap_x(phi_4, x):  # 计算a_bcd纠缠熵
    p_x = {}
    a = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    idx = a[x]
    # 建立p_x
    for key, value in phi_4.items():
        for key_1, value_1 in phi_4.items():
            ijkl = find_key_ijkl(key)
            ijkl_1 = find_key_ijkl(key_1)
            ijkl_x = copy.deepcopy(ijkl)
            ijkl_1_x = copy.deepcopy(ijkl_1)
            ijkl_x.pop(idx), ijkl_x.pop(0)
            ijkl_1_x.pop(idx), ijkl_1_x.pop(0)
            if ijkl_x == ijkl_1_x:
                new_key = ijkl[0] + ijkl[idx] + ' ' + ijkl_1[0] + ijkl_1[idx]
                inital = p_x.get(new_key, 0)
                p_x[new_key] = value * value_1.conjugate() + inital
    # 计算p_ax
    idx_matrix = {}
    ii = 0
    for key, value in p_x.items():
        idx = find_space(key)
        if key[:idx] == key[idx + 1:]:
            idx_matrix[key[:idx]] = ii
            ii += 1
    matrix = numpy.zeros((ii, ii))
    for key_1, value_1 in p_x.items():
        idx_1 = find_space(key_1)
        aa = idx_matrix[key_1[:idx_1]]
        bb = idx_matrix[key_1[idx_1 + 1:]]
        matrix[aa][bb] = value_1.real
    w, v = numpy.linalg.eig(matrix)
    Sum = 0
    for ww in w:
        if ww > 1e-8:
            Sum += -ww * numpy.log(ww)

    # print(Sum)
    return Sum


def comput_entropy(Seq):  # 输入序列，输出S
    ab, ac, ad, ae, af, bc, bd, be, bf, cd, ce, cf, de, df, ef = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    ABCD = [ab, ac, ad, ae, af, bc, bd, be, bf, cd, ce, cf, de, df, ef]
    create_initial_state(ABCD)
    for exp in Seq:
        Idx = find_function(exp)
        if Idx == 0:
            BS(exp, ABCD)
        elif Idx == 1:
            Do(exp, ABCD)
        elif Idx == 2:
            Re(exp, ABCD)
        elif Idx == 3:
            DP(exp, ABCD)
        else:
            OA(exp, ABCD)
    # 获得四光子状态
    phi_4 = {}
    idx_set = [9, 6, 5]
    for _ in range(3):
        set = ABCD[_]
        set_back = ABCD[idx_set[_]]
        # print(set,'\n',set_back)
        for key, value in set.items():
            for key_back, value_back in set_back.items():
                if _ == 0:
                    new_key = key + ' ' + key_back
                elif _ == 1:
                    key_space = find_space(key)
                    key_back_space = find_space(key_back)
                    new_key = key[:key_space + 1] + key_back[:key_back_space + 1] + key[key_space + 1:] + key_back[
                                                                                                          key_back_space:]
                else:
                    key_space = find_space(key)
                    new_key = key[:key_space + 1] + key_back + key[key_space:]
                # print(new_key)
                inital = phi_4.get(new_key, 0)
                phi_4[new_key] = inital + value * value_back
                # print(1)
    #查验是否有输出
    m = []
    for value in phi_4.values():
        m.append(value)
    MM = len(m)
    if m == list(numpy.zeros(MM)):
        return 0

    # 归一化phi_4
    sum_value = 0
    for key, value in phi_4.items():
        sum_value += abs(value) ** 2
    sum_value = numpy.sqrt(sum_value)
    for key, value in phi_4.items():
        phi_4[key] = value / sum_value
    # print(phi_4)
    # 计算纠缠熵和秩
    # 计算p_a,b,c,d
    p_a = comput_S_p_x(phi_4, 'a')
    p_b = comput_S_p_x(phi_4, 'b')
    p_c = comput_S_p_x(phi_4, 'c')
    p_d = comput_S_p_x(phi_4, 'd')
    # 计算p_ab,ac,ad
    p_ab = comput_S_ap_x(phi_4, 'b')
    p_ac = comput_S_ap_x(phi_4, 'c')
    p_ad = comput_S_ap_x(phi_4, 'd')
    S = p_a + p_b + p_c + p_d + p_ab + p_ac + p_ad
    return S


#example = ['BS(XXX,b,f)', 'BS(XXX,b,e)', 'BS(XXX,b,e)', 'BS(XXX,a,c)']

#S = comput_entropy(example)
#print(S)
