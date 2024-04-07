import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import chi2

def linear_fit(x, y):
    """
    对给定的数据点执行线性拟合。

    参数:
    - x (list): x值列表(自变量)。
    - y (list): y值列表(因变量)。

    返回:
    - delta (float): 线性拟合计算中使用的delta值。
    - slope (float): 拟合直线的斜率。
    - intercept (float): 拟合直线的截距。
    - slope_error (float): 斜率的标准误差。
    - intercept_error (float): 截距的标准误差。
    - R (float): 皮尔逊相关系数。
    """
    N = len(x)
    x_bar = sum(x) / N
    y_bar = sum(y) / N

    # 计算 delta
    delta = N * sum(x_i ** 2 for x_i in x) - sum(x) ** 2

    # 计算斜率和截距
    slope = (N * sum(x[i] * y[i] for i in range(N)) - sum(x) * sum(y)) / delta
    intercept = y_bar - slope * x_bar

    # 计算标准误差
    s_yx = math.sqrt(1 / (N - 2) * sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(N)))
    slope_error = math.sqrt(N * s_yx ** 2 / delta)
    intercept_error = math.sqrt(s_yx ** 2 * sum(x_i ** 2 for x_i in x) / delta)

    # 计算皮尔逊相关系数
    R = math.sqrt(1 - (N - 2) * (s_yx ** 2) / sum((y_i - y_bar) ** 2 for y_i in y)) * (-1 if slope < 0 else 1)

    return delta, slope, intercept, slope_error, intercept_error, R

def calculate_chi_square_residuals(x, y, y_pred, y_err):
    """
    计算卡方值和残差。

    参数：
    - x (array): x值数组（自变量）。
    - y (array): 观测值数组（因变量）。
    - y_pred (array): 拟合模型的预测值数组。
    - y_err (array): 观测值的误差数组。

    返回：
    - chi_square (float): 卡方值。
    - residuals (array): 残差数组。
    """
    x_arr = np.array(x)
    y_arr = np.array(y)
    y_pred_arr = np.array(y_pred)
    y_err_arr = np.array(y_err)
    
    residuals = (y_arr - y_pred_arr) / y_err_arr
    chi_square = np.sum(residuals ** 2)
    return chi_square, residuals

def calculate_degrees_of_freedom(x, num_params):
    """
    计算自由度。

    参数：
    - x (array): x值数组（自变量）。
    - num_params (int): 拟合模型的参数数量。

    返回：
    - degrees_of_freedom (int): 自由度。
    """
    return len(x) - num_params

def calculate_chi_square_probability(chi_square, degrees_of_freedom):
    """
    计算卡方概率。

    参数：
    - chi_square (float): 卡方值。
    - degrees_of_freedom (int): 自由度。

    返回：
    - p_value (float): 卡方概率。
    """
    p_value = 1 - chi2.cdf(chi_square, degrees_of_freedom)
    return p_value

def process_x(list_x,list_x_error):
    '''
    处理自变量取值以满足线性拟合条件，取决于实验本身的内容
    '''
    re1 = []; re2 = []
    for i in range(len(list_x)):
        x = list_x[i]; x_err = list_x_error[i]; x_err_percentage = x_err/x
        c = 286; c_error = 2 ; c_err_percentage = c_error / c
        new_x = 1/(x - c) 
        new_err_percentage = (x_err_percentage**2 + c_err_percentage**2)**0.5
        new_err = new_x * new_err_percentage
        re1.append(new_x)
        re2.append(new_err)
    return re1, re2

def zoom(list_x, k = 10): #由于个别情况下误差极其不合理，因此我们对其放大
    re = []
    for x in list_x:
        re.append(x*k)
    return re

def get_data():
    """
    获取用于测试线性拟合的样本数据。
    """
    # 样本数据
    x = [447.1, 471.3, 492.2, 501.6, 587.6, 667.8, 434.0, 486.1, 656.3]  
    y = [12.95, 11.88, 11.48, 11.34, 8.88, 7.50, 15.58, 12.16, 7.63]  

    # 测量误差
    x_err = [0.05] * 9  # x测量误差
    y_err = [0.02] * 9  # y测量误差

    #修正自变量
    x, x_err = process_x(x,x_err)

    x_err = zoom(x_err,k = 8)
    y_err = zoom(y_err,k = 10)


    title = 'Scale reading y repect to 1/(λ - λ0)'
    return x, x_err, y, y_err, title

# 获取样本数据
x, x_err, y, y_err, title = get_data()

# 执行线性拟合
delta, slope, intercept, slope_error, intercept_error, R = linear_fit(x, y)


# 绘图
x_start = min(x); x_start = min(x_start*0.9, x_start*1.1)
x_end = max(x); x_end = max(x_end*0.9,x_end*1.1)
y_start = min(y)
y_end = max(y)
x_vals = np.linspace(x_start - (x_end - x_start) * 0.05, x_end + (x_end - x_start) * 0.05, 100)

x_margin = (x_end - x_start) * 0.05  # 在x轴两端留出5%的空间
y_margin = (y_end - y_start) * 0.05  # 在y轴两端留出5%的空间

plt.scatter(x, y, color='tab:blue', s = 5)
plt.xlim(x_start - x_margin, x_end + x_margin)
plt.ylim(y_start - y_margin, y_end + y_margin)  # 根据数据调整y轴范围

plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', markersize=5, capsize=3)

# 绘制线性函数及其误差范围
plt.plot(x_vals, intercept + slope * x_vals, lw=2, color='tab:blue')
plt.plot(x_vals, (intercept + intercept_error) + (slope - slope_error) * x_vals, '--', lw=1, color='tab:green')
plt.plot(x_vals, (intercept - intercept_error) + (slope + slope_error) * x_vals, '--', lw=1, color='tab:green')

plt.title(title)
plt.xlabel('1/(λ - λ0)')
plt.ylabel('y')
plt.legend(['_a', 'Estimated Trendline', 'Trendline Boundaries'])

# 打印皮尔逊相关系数和线性函数，包含误差
print(f'Pearson Correlation Coefficient R = {R}')
print(f'Linear Function: y = {slope:.4f} ± {slope_error:.4f} x + {intercept:.4f} ± {intercept_error:.4f}')

# 计算预测值
y_pred = [intercept + slope * xi for xi in x]

# 计算卡方值和残差
chi_square, residuals = calculate_chi_square_residuals(x, y, y_pred, y_err)

# 计算自由度
degrees_of_freedom = calculate_degrees_of_freedom(x, num_params=2)  # 假设拟合模型是二次的

# 计算卡方概率
p_value = calculate_chi_square_probability(chi_square, degrees_of_freedom)

print(f"卡方值：{chi_square}")
print(f"自由度：{degrees_of_freedom}")
print(f"卡方概率：{p_value}")

# 在图像上添加线性函数及其误差范围的文本标注
plt.text(0.0023, 14, f'y = ({slope:.4f} ± {slope_error:.4f})x + ({intercept:.4f} ± {intercept_error:.4f})', ha='left')
plt.text(0.0023, 13.5, f'Pearson Correlation Coefficient R = {R}', ha='left')
plt.text(0.0023, 13, f'Chi squared: {chi_square}', ha='left')
plt.text(0.0023, 12.5, f'Degrees of freedom: {degrees_of_freedom}', ha='left')
plt.text(0.0023, 12, f'Normalized Chi-Square: {chi_square/degrees_of_freedom}', ha='left')
plt.text(0.0023, 11.5, f'Chi squared probability: {p_value}', ha='left')

plt.show()

def residual_plot(x, y, y_pred):
    """
    绘制残差图。

    参数:
    - x (list): x值列表(自变量)。
    - y (list): y值列表(因变量)。
    - y_pred (list): 根据线性拟合计算得到的y值列表(拟合值)。
    """
    residuals = [y[i] - y_pred[i] for i in range(len(x))]
    plt.scatter(x, residuals, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.errorbar(x, residuals, xerr = x_err, yerr=y_err, fmt='o', color='red')
    plt.xlabel('1/(λ - λ0)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

# 绘制残差图
residual_plot(x, y, y_pred)
