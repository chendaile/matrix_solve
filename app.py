# app.py
from flask import Flask, request, render_template
import numpy as np
import sympy as sp
from io import StringIO

app = Flask(__name__)

def get_index(a_array, except_index):
    for i in range(a_array.shape[0]):
        if i in except_index:
            continue
        if a_array[i] != 0:
            return i
    return -1

def reshape(matrix, volumn_index):
    matrix_copy = matrix.copy()
    for i in range(matrix_copy.shape[0]):
        if matrix_copy[i, volumn_index] != 0:
            matrix_copy[i, :] = matrix_copy[i, :] / matrix_copy[i, volumn_index]
    return matrix_copy

def delete_to_zero(matrix, theone_index, volumn_index):
    theone_row = matrix[theone_index, :]
    for i in range(matrix.shape[0]):
        if i == theone_index or matrix[i, volumn_index] == 0:
            continue
        matrix[i, :] -= theone_row * matrix[i, volumn_index]
    return matrix

def process_matrix(matrix_input, b_input):
    try:
        matrix = np.genfromtxt(StringIO(matrix_input), delimiter=' ')
        b = np.genfromtxt(StringIO(b_input), delimiter=' ').reshape(-1, 1)
    except Exception as e:
        return None, f"输入格式错误: {str(e)}"
    
    if matrix.shape[0] != b.shape[0]:
        return None, "系数矩阵行数与常数项向量长度不匹配"
    
    augmented = np.hstack((matrix, b)).astype(object)
    except_index = []
    
    # 高斯消元主逻辑
    for i in range(augmented.shape[1]-1):
        theone_index = get_index(augmented[:, i], except_index)
        if theone_index == -1:
            continue
        except_index.append(theone_index)
        augmented = reshape(augmented, i)
        augmented = delete_to_zero(augmented, theone_index, i)
    
    # 处理自由变量
    J = []
    for i in range(augmented.shape[1]-1):
        for j in range(augmented.shape[0]):
            if augmented[j, i] != 0 and j not in J:
                J.append(j)
                augmented[j, :] = augmented[j, :] / augmented[j, i]
                break
    
    # 检查无解情况
    non_solution = False
    solution = []
    for i in range(augmented.shape[0]):
        non_zero_list = np.where(augmented[i, :-1] != 0)[0]
        if len(non_zero_list) == 0 and augmented[i, -1] != 0:
            non_solution = True
            break
            
        # 构建解集
        if len(non_zero_list) > 0:
            var_index = non_zero_list[0]
            expr = augmented[i, -1]
            for col in non_zero_list[1:]:
                expr -= augmented[i, col] * sp.Symbol(f'x_{col+1}')
                augmented[i, col] = 0
            solution.append({
                'variable': f'x_{var_index+1}',
                'expression': sp.latex(sp.simplify(expr))
            })
    
    return {
        'matrix': augmented,
        'non_solution': non_solution,
        'solution': solution
    }, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        matrix = request.form['matrix'].strip()
        b_vector = request.form['b_vector'].strip()
        
        if not matrix or not b_vector:
            return render_template('index.html', error="输入不能为空")
        
        result, error = process_matrix(matrix, b_vector)
        if error:
            return render_template('index.html', error=error)
            
        return render_template('result.html', 
                             matrix=result['matrix'],
                             non_solution=result['non_solution'],
                             solution=result['solution'])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # ← 关键修改这里