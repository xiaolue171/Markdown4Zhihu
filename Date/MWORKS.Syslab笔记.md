# 基础知识
## 变量命名

和其他编程语言差不多.

## 标点符号

- `;`: 语句结束不需要分号, 换行即可. 区别在于加了分号的语句不会立即显示执行结果;
- `#`: 井号后面的是单行注释;
- `#= ... =#`: 井号-等号中间的是多行注释.

## 控制结构

1.  `for`循环

	```julia
	for x in array # 也可以写for x = array
		expression
	end
	```

2. `while`循环

	```julia
	while expression
		statement
	end
	```

3. `if-else-end`选择结构

	```julia
	if expression_1
		statement_1
	elseif expression_2
		statement_2
	else
		statement_3
	end
	```

## 计算
### 运算符

- `+, -, *, /, \, ^`: 两个数之间的加, 减, 乘, 除以, 除, 幂运算;
- `.+, .-, .*, ./, .\, .^`: 对矩阵/向量的每个元素进行的加, 减, 乘, 除以, 除, 幂运算;
- `<, <=, >, >=, ==, !=`: 在前面加`.`后可以用于矩阵
- `||, &`: 或, 且
- `!==`: 恒不等于

### 常数

| 符号    | 含义   | 输入       |
| ----- | ---- | -------- |
| pi    | 圆周率  | `pi`     |
| $\pi$ | 圆周率  | `\pi`    |
| $e$   | 自然常数 | `\euler` |
| im    | 虚数单位 | `im`     |

更多常数可以使用`Base.MathConstants`包获得.

### 数学函数

1. 三角函数
	- `sin, cos, tan, cot, sec, csc`
	- `asin, acos, atan, acot, asec, acsc`
	- `sinh, cosh`
2. 指数函数
	- `exp`
	- `log`: 自然对数$ln$
	- `log10`: 以10为底的对数$lg$
	- `log2`: 以2为底的对数
	- `pow2`: 2的幂
	- `sqrt`: 开方
3. 取整函数
	- `ceil`: 向上取整
	- `trunc`: 向0取整
	- `floor`: 向下取整

其他还有
- `all`函数
	```julia
	all(A); # 判断数组或向量A中的元素值是否全为true;
	all(p, A); # 判断A中的元素是否全满足条件p;
	```
- `factorial`函数--求阶乘

## 编写函数
### 基本方法

```julia
function function_name(args)
	expression_1
	...
	expression_n
end
# 函数的值是最后一个表达式, 即expression_n的值
# e.g.
function f(x, y)
	return x + y # 与x + y等价
end
```

也可以这样创建函数:

```julia
f = x -> expression;
f = (x, y, ...) -> expression;
```

或者这样:

```julia
f(x, y, ...) = expression;
```

### 编写脚本文件

首先编写函数文件function.jl

```julia
# function.jl
function myadd(x, y)
	return x+y
end
```

ctrl+F5运行脚本, 在终端中使用

```julia
julia> myadd(3, 5)
8
```

或者在其他文件中引用该脚本

```julia
# main.jl
include("function.jl"); # 这里假设main.jl和function.jl在同一目录下
myadd(3, 5) # 8
```

# 基本数据结构
## 向量
### 创建

```julia
vec = [1, 2, 3, 4, 5];
vec = [i for i in 1:5];
```

### 拼接

```julia
vec1 = [1, 2, 3, 4, 5];
vec2 = [6, 7, 8, 9, 10];

vec = [vec1; vec2];
vec = [1, 2, 3, 4, 5; vec2];
# vec == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

## 数组

注: Julia中的数组是从1开始编号的, 而不是从0开始.

### 创建
1. 简单数组

	```julia
	x = [1 2 3 4]; # 用空格分隔相邻元素
	```

2. 等距数组

	```julia
	# 基本方法
	x = first: last; # 创建从first开始, 步长为1, 到last结束的行向量, 维数为(last-first+1)
	x = first: increment: last; # 可以指定步长

	# LinRange
	x = LinRange(first, last, n); # 把firse到last区间n-1等分, 获得一共n个点, 形成一个n维行向量
	```

## 矩阵
### 创建

1. 简单矩阵

	```julia
	x = [1 2 3 4; 5 6 7 8]; # 用分号区分相邻行
	x = [
		1 2 3 4
		5 6 7 8
	]; # 也可以手动分行
	```

2. 特殊矩阵

	```julia
	x = []; # 创建一个空矩阵
	x = zeros(m, n); # 创建一个m*n的零矩阵
	x = ones(m, n); # 创建一个m*n的1矩阵
	x = eye(n); # 创建一个n阶单位矩阵
	x = eye(m, n); # 创建一个m*n的单位矩阵
	x = rand(m, n); # 创建一个m*n的随机矩阵, 其元素在0到1之间
	```

3. 对角矩阵

	```julia
	A = diagm(v); # 生成主对角线是向量v的对角矩阵.
	A = diagm(n=>v); # 生成从主对角线向右数第n条对角线为向量v的矩阵
	```

4. 拼接矩阵

	```julia
	[A B]; # 左右拼接A, B构成新矩阵
	[A; B]; # 上下拼接A, B构成新矩阵
	```

5. 子矩阵

	```julia
	# 只需要记住基本公式是A[m1:stepm:m2, n1:stepn:n2]
	A[r, :]; # 表示提取A的第r行, 每一列
	A[:]; # 依次提取A的每一列组成长列向量
	A[i:j, m:n]; # 提取A的i到j行, m到n列作为子矩阵
	A[j:-1:i, ...]; # 逆序提取
	A[1:2:end, :]; # 提取A的所有奇数行, 所有列

	# 特殊子矩阵提取
	reverse(A, dims=2); # 翻转A的第二个维度, 即每一行
	diag(A, k); # 提取A的第k条对角线为列向量
	tril(A); # 提取A的下三角
	triu(A); # 提取A的上三角
	```

### 运算

```julia
Y = X' # 转置, 数组也可以转置
det(A); # 计算行列式
inv(A); # 计算逆矩阵
A/B; # 计算X*A=B的解
A\B; # 计算A*X=B的解
rank(A); # 计算矩阵A的秩
r, p = rref(A); # 计算矩阵的简化阶梯形r和主元所在列p
```

# 作图
## 二维作图函数 

1. `plot`函数
   
	```julia
	# plot()+取点作图
	x = LinRange(...);
	y = f.(x);
	plot(x, y, fmt); # 可以在fmt中指定线形之类的信息
	plot(x1, y1, "S1", x2, y2, "S2", ...);
	```

2. `ezplot`函数

	```julia
	ezplot(x -> f(x), [a,  b]);
	ezplot((x, y) -> f(x, y), [xmin, xmax, ymin, ymax]); # 隐函数
	ezplot(t -> x(t), t -> y(t), [tmin, tmax]); # 参数方程
	```

3. `fplot`函数

	```julia
	fplot(f, [xmin, xmax]);
	fplot([f, g, h, ...], [xmin, xmax]); # 一张图上可以画多个函数
	```

## 作图修饰
### 基本线形和颜色

| 符号  | 颜色  | 符号  | 线形    |
| --- | --- | --- | ----- |
| y   | 黄色  | .   | 空心圆点  |
| r   | 红色  | o   | 空心圆圈  |
| g   | 绿色  | ^   | 空心三角形 |
| b   | 蓝色  | *   | 空心五角星 |
| k   | 黑色  | -   | 实线    |
|     |     | --  | 虚线    |
|     |     | :   | 点线    |
|     |     | -.  | 点画线   |

### 格栅, 图例与注释

```julia
grid("on");
grid("off");

xlabel("string"); # 将string作为轴标题
ylabel("string");
title("string");

gtext("string"); # 在鼠标点击的地方加字符串
```

### 坐标轴操作: `axis`函数

```julia
axis(style);
```

style是字符串, 可以取包括但不限于以下值:

| 值        | 说明             |
| -------- | -------------- |
| "on/off" | 打开/关闭坐标轴       |
| "auto"   | 自动缩放           |
| "square" | 显示为正方形         |
| "equal"  | 坐标轴长度单位设置为相等   |
| "normal" | 关闭equal和square |

- 一个常用命令
	```julia
	axis([a1, a2, a1, a2]);
	axis("square");
	```

### 图形保持

```julia
hold("on");
hold("off");
figure(h); # 打开或切换到第h个窗口
subplot(m, n, k); # 划分为m*n个窗口, 激活第k块
```

## 极坐标绘图

```julia
polarplot(theta, r, "s"); # 以theta为角度, r为极半径, s为线形画极坐标图形
```

## 空间绘图
### 一些重要的函数

```julia
# meshgrid() -- 绘制网格
X, Y = meshgrid2(x); # x是一个n维向量, 该函数生成两个n*n矩阵, 其中X矩阵每一行都是x向量, Y矩阵每一列都是x向量

# peaks() -- 绘制用于演示的示例曲面
Z = peaks(); # 返回一个49*49矩阵
Z = peaks(n); # 返回一个n*n矩阵
Z = peaks(X, Y); # 返回一个使用X, Y矩阵生成的矩阵
...
```

### 空间曲线

1. 单条曲线

	```julia
	plot3(x, y ,z, "s"); # x, y, z是一些点的坐标
	fplot3(x, y, z, [tmin, tmax]); # 其中x, y, z是关于t的函数
	```

2. 多条曲线

	```julia
	plot3(x, y, z); # x, y, z都是m*n矩阵, 每一列代表一条曲线上
	```

### 空间曲面

1. `surf`函数--生成曲面

	```julia
	surf(X, Y, Z); # X, Y, Z是三个相同大小的矩阵, 它们的(i, j)-元分别构成一个点的x, y, z轴上的坐标
	surf(X, Y, Z, alpha=0.5); # 也可以加参数绘制半透明曲面
	```

2. `mesh`函数--生成网格曲面

	```julia
	mesh(X, Y, Z); # 原理同surf()函数
	```

3. `meshz`函数--在网格周围绘制curtain图

	```julia
	meshz(X, Y, Z); # 原理同surf()函数, 但是这个图像看起来挺漂亮的
	```

### 其他函数

```julia
scatter(x, y); # 绘制二维散点图
scatter3(x, y, z); # 绘制三维散点图
surface(X, Y, Z); # 基本曲面图
surfc(X, Y, Z); # 曲面图下的等高线图
meshc(X, Y, Z); # 具有基本等高线的网格图
fsurf(f, [xmin, xmax, ymin, ymax]); # 在指定范围内绘制曲面图
```

# 导数与偏导数

注: 需要使用`TySymbolicMath`

## 求导数与偏导数

1.  `derivative`函数--求导

	```julia
	derivative(f, x); # 求f的对x的一阶导数或偏导数, 其中f是表达式
	```

2. `jacobian`函数--求jacobian矩阵

	```julia
	jacobian([f, g, h], [x, y, z]); # 其中f, g, h是关于x, y, z的三元函数
	```

输出为

$$
\begin{pmatrix}
\frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial z} \\
\frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} & \frac{\partial g}{\partial z} \\
\frac{\partial h}{\partial x} & \frac{\partial h}{\partial y} & \frac{\partial h}{\partial z}
\end{pmatrix}
$$

eg. 求$u=\sqrt{x^2+y^2+z^2}$关于$x$和$y$的偏导数.

```julia
using TySymbolicMath
@variables x y z

u = (x^2+y^2+z^2)^(1/2);
a = jacobian([u], [x, y]);
# a为1×2 Matrix{Num}:
# x / ((x^2 + y^2 + z^2)^0.5)  y / ((x^2 + y^2 + z^2)^0.5)
```

## 极值和最值

1.  `fminbnd`函数--求区间内局部最小值

	```julia
	x, fval = fminbnd(f, xmin, xmax); # 返回f在区间[xmin, xmax]上的一个极小值, 其中f是一个函数而非表达式
	```

2. `fminsearch`函数--求多元函数局部最小值

	```julia
	x, fval = fminsearch(f, x0); # 寻找在x0附近的局部最小值, 其中x0可以是点或者向量
	```

## 函数零点

```julia
x, fval = fzero(f, x0); # 求f在x0附近的一个零点, 并求f在零点处的值(一个很小的数), 其中x0可以是点或者向量
```

## 反函数

```julia
g = finverse(f); # 求f的反函数
g = finverse(f, x); # 求f关于x的反函数
```

# 积分
## 求和

```julia
s = sum(A); # 对A中所有元素求和, 其中A可以是数组或矩阵
s = sum(A, dims=n); # 对A的第n个维度求和. 例如, 对于二维矩阵, dims=1代表对列求和, dims=2代表对行求和
s = sum(f, x): # 对x中的元素取f函数值之后求和
```

## 不定积分

```julia
F = int(expr); # 求expr的不定积分, expr必须是一个表达式
F = int(expr, x); # 求expr关于x的不定积分

# int函数也可以用于求定积分, 而且xmin, xmax可以为符号变量
F = int(expr, xmin, xmax);
F = int(expr, x, xmin, xmax);
```

## 定积分

注: 以下两个函数中`xmin`, `xmax`必须为常数.

1. `quad()` 函数--Simpson方法求积分

	```julia
	q, e = quad(f, xmin, xmax); # 求积分q并返回误差e, 其中f必须为函数名

	# e.g. 求sin(x)/x从0到pi的积分
	f = x -> sin(x)/x;
	q, e = quad(f, 0, pi);
	```

2. `integral()`系列函数求积分

	```julia
	q, e = integral(f, xmin, xmax);
	q, e = integral2(f, xmin, xmax, ymin, ymax); # 二重积分
	q, e = integral3(f, ...); # 三重积分
	```

e.g. 求
$$
\int_{0}^{1}\,dx\int_{2x}^{x^2+1} xy\,dy
$$

```julia
f(x, y) = x*y;
q, e, = integral2(f, 0, 1, 2*x, x^2+1);
```

## `sympy`库的使用

```julia
using PyCall
sympy = pyimport("sympy"); # 引入sympy库
x = sympy.symbols("x"); # 定义符号变量
F = sympy.integrate(f, x); # 求f关于x的不定积分
F = sympy.integrate(f, (x, xmin, xmax)); # 求f关于x从xmin到xmax的定积分, 上下限可以为符号变量
df = sympy.diff(f, x); # 求f关于x的导数
```

# 数据处理
## 插值

1. `LagrangeInterp`--Lagrange插值法

	```julia
	y = Lagrange(x0, y0, x); # 对点列(x0, y0)做插值, 并返回点列x处的y值
	```

2. `interp1`--分段插值

	```julia
	y1 = interp1(x0, y0, x); # 分段线性插值: 直线连接相邻点, 不光滑
	y2 = interp1(x0, y0, x, "spline"); # 分段三次插值: 三次多项式连接相邻点, 较光滑
	y3 = interp1(x0, y0, x, "pchip"); # 三次样条插值: 三次多项式连接相邻点, 且强制全局二阶导数连续, 光滑
	y4 = ... # 还有多种插值方法
	```

3. `interp2`--二维插值

	```julia
	Vq = interp2(X, Y, V, Xq, Yq); # 对网格点计算插值并查询, 需要X, Y, V是矩阵
	... # 还有多种插值方法
	vq = griddata(x, y, v, xq, yq); # 对非网格散点计算插值并查询, 需要x, y, v是向量
	```

## 拟合

1. `ployfit`函数--一元多项式拟合
   
	```julia
	p, = ployfit(x, y, n); # 返回一个n次拟合函数的系数, 其中x, y是两个维数相同的向量
	```

2. `fit`函数--二元多项式拟合
	
	需要使用`TyCurveFitting`包

	```julia
	fitobject = fit("ploymn", [x, y], z); # 返回一个拟合函数的系数, 其中x, y, z是两个向量, m, n分别表示拟合函数关于x, y的次数
	```

# 常微分方程的数值解
## `ode`系列函数
### 非刚性微分方程(组)

- `ode23()`
- `ode45()`: 可以满足大多数需求
- `ode78()`
- `ode113()`

### 刚性微分方程(组)
- `ode15s()`: `ode45()`失效时可试用
- `ode23s()`
- `ode23t()`
- `ode23tb()`

### 使用方法

```julia
t, y = solver(odefun, tspan, y0);
# solver是一个ode函数名 
# odefun是一个微分方程(组)表达式: dy/dt = f(t, y)
# tspan取[t0 tf], 是t的积分范围, t0为初始点
# y0是初始条件(标量或列向量), 即y(t0)=y0
# t是一个列向量[t1; t2; ...], 是由函数算法生成的一些取值点
# y是一个列向量或矩阵, 其中元素yn1, yn2, ...分别表示第n个方程的解在t1, t2, ...处的取值
```

通过绘制$y-t$图像, 可以得出常微分方程的解的图像$y=y(t)$.

## 高阶常微分方程

可以化为一阶微分方程组来求解.

e.g. 求$y''(t)=2(1-y^2)y'-y, y(0)=1, y'(0)=0$的解.

将方程转化为一阶方程组:
$$
\begin{cases}
y_{1}'(t)=y_{2}(t), y_{1}(0)=1\\
y_{2}'(t)=2(1-y_{1}^2)y_{2}-y_{1}, y_{2}(0)=0
\end{cases}
$$
之后代码如下:

```julia
function f(t, y)
	return [y[2]; 2*(1-y[1]^2)*y[2]-y[1]];
end

t, y = ode45(f, [0 15], [1;0]);
plot(t, y[:,1], "o", t, y[:,2], "*");
```

# 线性代数
## 多项式
### 多项式的表达和求根

1. `poly2sym`函数--创建多项式

	```julia
	using TySymbolicMath
	poly2sym(p); # 返回以向量p为系数的多项式, 从左到右, 对应幂次从高到低, 默认以x为符号变量
	poly2sym(p, v); # 返回以v作为符号变量的多项式
	```

2. `roots`函数--求多项式复数根

	```julia
	roots(p); # 求以向量p为系数的多项式的所有复数根, 包括重根

	# e.g. 求x^2+2*x+3=0的根
	roots([1,2,3]);
	```

3. `polyval`函数--求多项式的值

	```julia
	polyval(p, a); # 求以向量p为系数的多项式, 取变量值为a时的值
	```

### 多项式的运算

1.  `conv`函数--多项式相乘

	```julia
	conv(p1,p2); # 求系数为向量p1, p2的多项式相乘后的系数向量
	```

2. `deconv`函数--多项式相除

	```julia
	a, b = deconv(p1, p2); # 计算系数为向量p1的多项式除以p2的多项式, a是商式的系数向量, b是余式的系数向量
	```

3. `collect`函数--合并同类项

	```julia
	collect(f); # 对符号多项式f合并同类项
	```

4. `expand`函数--展开多项式

	```julia
	expand(f); # 展开符号多项式f
	```

5. `factor`函数--因式分解

	```julia
	factor(f); # 对符号多项式f因式分解
	```

6. `residue`函数--多项式除法的部分分式展开

	```julia
	a, b, r = residue(p,q); # 将系数向量为p, q的多项式相除的结果展开为部分分式之和, 其中a为部分分式分子向量, b为部分分式分母向量, r为余式多项式向量
	```

## 向量
### 向量组的线性相关性

将需要判断的向量组中的向量按列排成矩阵求秩.

```julia
a1 = [1 2 2 3];
a2 = [1 4 -3 6];
a3 = [-2 -6 1 -9];
a4 = [1 4 -1 7];
a5 = [4 8 2 9];
A = [a1' a2' a3' a4' a5'];
rank(A);
```

### 向量组的极大无关组

将向量组中的向量按排成矩阵, 求简化阶梯形的主元所在列, 该列即为极大无关组的向量.
```julia
r, p = rref(A);
p # p指出的列为极大无关组向量
```

## 线性方程组的解
### 齐次线性方程组的解

`null`函数用于求矩阵的零空间, 这里用于解齐次线性方程组.

```julia
B = null(A); # B的列向量是以A为系数矩阵的齐次线性方程组的基础解系
B = null(A, "r"); # 把基础解系表示为有理数形式
```

### 非齐次线性方程组的解
`linsolve()`函数--求一个解

```julia
linsolve(A, b); # 求AX=b的一个解, 若无解则求最小二乘解, A可以是符号矩阵, 必须行满秩
```

## 特征值和特征向量 

`eigvals和eigvecs`函数--求矩阵特征值和特征向量

```julia
eigvals(A); # 求矩阵A的特征值, 返回一个向量
eigvecs(A); # 求矩阵A的特征向量, 返回一个矩阵, 列向量作为特征向量, 与特征值向量一一对应
```

## Vandermonde行列式

```julia
A = vander(v); # 返回Vandermonde行列式, 使其列向量是向量v的幂, 从右到左升幂
```

# 概率与统计

需要使用`TyStatistics`

## 古典概率

- 随机变量的分布函数
- 离散型随机变量的分布
- 连续性随机变量的概率密度函数

## 随机变量与概率分布

- pdf--概率密度函数; cdf--分布函数

1. `bino`--二项分布

	$$
	P(X=k) = C^k_{n}p^k(1-p)^{n-k}
	$$

	```julia
	P = binopdf(k, n, p); # 在X~B(n, p)的二项分布中X=k时的概率密度函数值
	P = binocdf(k, n, p); # 在X~B(n, p)的二项分布中X=k时的分布函数值
	# k可以为向量, 函数可以取点运算
	```

2. `geo`--几何分布
3. `hyge`--超几何分布
4. `poiss`--泊松分布

	$$
	P(X=k) = \frac{\lambda^k}{k!}e^{-\lambda}
	$$

	```julia
	P = poisspdf(k, lambda);
	P = poisscdf(k, lambda);
	```

5. `unid`--离散均匀分布

	```julia
	P = unidpdf(k, N);
	P = unidcdf(k, N);
	```

6. `unif`--均匀分布

	```julia
	P = unifpdf(k, a, b); # 区间a, b上的均匀分布
	P = unifcdf(...)
	```

7. `exp`--指数分布

	$$
	f(x) = \frac{1}{\theta}e^{-x/\theta}
	$$

	```julia
	P = exppdf(k, theta);
	P = expcdf(k, theta);
	```

7. `norm`--正态分布

	```julia
	P = normpdf(k, mu, sigma);
	P = normcdf(k, mu, sigma);
	```

## 逆累积分部函数icdf

- **逆累积分部函数**返回给定概率条件下自变量的临界值, 实际上是分布函数的反函数: 输入概率$p$, 输出分布函数为$p$时对应$x$的值.

函数基本用法为`nameinv(p, parameters)`, 其中name为概率函数的名称, 如bino, exp等等, parameters是对应函数的参数, 如bino对应的参数就是n, p, 参数顺序与概率密度函数和分布函数保持一致.

# 随机数与积分

## 随机数的产生

`pd = fun(...); r = random(pd, m, n);`
其中`fun(...)`是相应概率分布函数的名称与参数, 参数顺序与上文保持一致. 后一行代码按照前述函数生成一个m行n列的随机数矩阵.

其次, 也可以使用函数`-rnd(..., m, n)`直接生成m行n列的随机数.

常用五种概率分布如下

| fun(...)           | -rnd(..., m, n)          | 分布类型 |
| ------------------ | ------------------------ | ---- |
| Normal(mu, sigma)  | normrnd(mu, sigma, m, n) | 正态分布 |
| Poisson(lambda)    | poissrnd(lambda, m, n)   | 泊松分布 |
| Binomial(n, p)     | binornd(n, p, m, n)      | 二项分布 |
| Uniform(a, b)      | unifrnd(a, b, m, n)      | 均匀分布 |
| Exponential(theta) | exprnd(theta, m, n)      | 指数分布 |

## 其他随机数的产生

- **定理**: 若$X$的分布函数为$F(x)=P(X \leqslant x)$, 则$F(x)$~$U(0,1)$.

构造分布函数为$F(x)$的随机数的方法如下:
1. 取$U(0, 1)$的随机数$U_i$;
2. 取$X_i=F^{-1}(U_i)$, 则$X_i$服从$F(x)$分布.

### 直接抽样法

e.g. 生成服从
$$
F\left( x \right) = \left\{ \begin{gathered}
    \frac{1}{2}{e^x},x \leqslant 0 \\
   1 - \frac{1}{2}{e^{ - x}},x > 0 \\ 
\end{gathered} \right.
$$

的随机数.

```julia
function F(n)
	y = unifrnd(0, 1, 1, n);
	for i in 1:n
		if y[i] > 1/2
			y[i] = -log(2-2*y[i])
		else
			y[i] = log(2*y[i])
		end
	return y
end
```

### 离散分布的直接抽样法

设分布律为$P(x=x_i)=p_i$, 则其分布函数为
$$
F\left( x \right) = \left\{ \begin{gathered}
    0,x < {x_i} \\
   \sum\limits_{j = 1}^i {{p_j}} ,{x_i} \leqslant x < {x_{i + 1}} \\ 
\end{gathered} \right.
$$
那么只需生成均匀分布随机数$R$~$U(0,1)$, 再按照
$$
X = \left\{ \begin{gathered}
    {x_i},F\left( {{x_i}} \right) < R \leqslant F\left( {{x_i}} \right) \\
   {x_1},R \leqslant F\left( {{x_1}} \right) \\ 
\end{gathered} \right.
$$
即可生成$X$~$F(x)$.

## 随机模拟法计算数值积分

- 例如, 要求某函数在区间上的最值, 只需要在这个区间上取大量均匀分布的随机数, 求出这些点处的最值即可.
- 相比于均匀分割区间来取点, 随机模拟方法多次运行可以得到不同结果, 准确度更高.

### 随机模拟方法求定积分

- 区间上的定积分, 数值上等于函数与坐标轴围成的曲边梯形的面积. 那么只需要取以区间为长的一个矩形, 在这个矩形内大量随机投点, 通过点落在曲边梯形内的概率来求定积分即可.
- 同样的思路, 计算二重积分, 只要在三维空间内随机投点即可.

### 计算几何图形面积

- 按照和求定积分相似的思路, 事实上我们可以求出任何由曲线围成的面积.