# app.py
from flask import Flask, request, render_template
import numpy as np
import sympy as sp
from io import StringIO
from gauss_elimination import perform_gaussian_elimination
import jinja2
app = Flask(__name__)

# Custom Jinja2 test for 'symbol'
def is_symbol(value):
    return isinstance(value, str) and not value.replace('.', '', 1).isdigit()
app.jinja_env.tests['symbol'] = is_symbol

# 新增的矩阵求逆处理函数
def process_inverse(matrix_input):
    try:
        input_matrix = np.genfromtxt(StringIO(matrix_input), delimiter=' ')
    except Exception as e:
        return None, f"矩阵格式错误: {str(e)}"
    
    #build the matrix
    Ax, Ay = input_matrix.shape
    build_matrix = np.zeros((Ay**2, Ax * Ay))
    b_matrix = np.zeros((Ay**2, 1))
    for i in range(Ay):
        for j in range(Ay):
            b_matrix[Ay * i + j] = 1 if i == j else 0
            for k in range(Ax):
                build_matrix[Ay * i + j, i * Ax + k] = input_matrix[k, j]

    #perform the gaussian elimination
    reverse_matrix = np.zeros((Ay, Ax)).astype(object)
    augmented_matrix, no_solve = perform_gaussian_elimination(build_matrix, b_matrix)
    if no_solve:
        return None, "矩阵不可逆"
    else:
        solve = {'变量': [], '表达式': []}
        for col in range(augmented_matrix.shape[1]-1):
            for row in range(augmented_matrix.shape[0]):
                i = col // Ax
                k = col % Ax
                reverse_matrix[i, k] = sp.Symbol(f'x_{col+1}')
                if augmented_matrix[row, col] == 1:
                    reverse_matrix[i, k] = augmented_matrix[row, -1]
                    solve['变量'].append(f'x_{i}{k}')
                    solve['表达式'].append(augmented_matrix[row, -1])
                    break
        return reverse_matrix, None

# 原有的线性方程组处理函数
def process_matrix(matrix_input, b_input):
    try:
        matrix = np.genfromtxt(StringIO(matrix_input), delimiter=' ')
        b = np.genfromtxt(StringIO(b_input), delimiter=' ').reshape(-1, 1)
    except Exception as e:
        return None, f"输入格式错误: {str(e)}"
    
    if matrix.shape[0] != b.shape[0]:
        return None, "系数矩阵行数与常数项向量长度不匹配"
    
    augmented_matrix, non_solution = perform_gaussian_elimination(matrix, b)

    if non_solution:
        return None, "方程组无解"
    else:
        # 构建解的输出
        solution = {'变量': [], '表达式': []}
        for col in range(augmented_matrix.shape[1]-1):
            for row in range(augmented_matrix.shape[0]):
                if augmented_matrix[row, col] == 1:
                    solution['变量'].append(f'x_{col+1}')
                    solution['表达式'].append(augmented_matrix[row, -1])
                    break

    return solution, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'b_vector' in request.form:  # 检测方程组求解表单
            return handle_linear_system()
        elif 'inverse_matrix' in request.form:  # 检测矩阵求逆表单
            return handle_matrix_inverse()
    
    return render_template('index.html')

def handle_linear_system():
    matrix = request.form['matrix'].strip()
    b_vector = request.form['b_vector'].strip()
    
    result, error = process_matrix(matrix, b_vector)
    if error:
        return render_template('index.html', error=error)
        
    return render_template('solution_result.html', 
                         solution=result)

def handle_matrix_inverse():
    matrix_input = request.form['inverse_matrix'].strip()
    
    inverse, error = process_inverse(matrix_input)
    if error:
        return render_template('index.html', error=error)
    
    return render_template('inverse_result.html', 
                         matrix=matrix_input,
                         inverse=inverse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)