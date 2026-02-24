---
title: "Java 基础知识"
date: '2025-08-14T15:22:11+08:00'
authors: [Xilyfe]
series: ["Java 技巧"]
tags: ["编程语言"]
--- 
 
# Java SE
## 类型转换

> 表达式结果的类型=最高类型
```
byte,short,char -> int -> long -> float -> double
		   char -> int

```

> byte,short,char直接转换为int类型
```java
byte i = 1;
short j = 30;
byte i = 2 * i; // error
short k = i + j; // error
int k = i + j; // correct
```

> 只要参与运算，就会提升到int类型
```java
byte i = 1;
byte j = (byte)2 * (byte)i; // error
byte j = (byte)(2 * i);
```

> 字符操作
```java
char random_alpha = (char)('a' + rand.nextInt(25));
```
## 运算符

> 两个整数相除还是整数

```java
int i = 5;
int j = 2;
float k = i / j; // k = 2
float k = (float)i / j; // k = 2.5
float k = 1.0 * i / j; // k = 2.5
```

> 赋值运算符带强制转换

```java
double a = 10;
int b = 30;
a += b; // a = (double)(a + b)
```
- 假如`i`和`j`都是byte类型，那么`i += j`与`i = i + j`结果不同
- `i += j`中`j`没有参与运算，还是byte类型
- `i = i + j`中，`i`和`j`都参与了运算，提升到int类型，int->byte需要强制类型转换

> &不管前面是否为false，全部statement都会执行；&&若遇到false，后面则不执行

## 分支控制
### Switch

1. 表达式类型只能是byte、short、int、char，JDK5支持枚举，JDK7支持String
```java
long variable = 100;
switch(variable){
	// error
}
```

2. 标签
```java
OUT:
for (int i = 0; i < 100; i++){
	for (j = i; j < 100; j++){
		if (j == 70){
			continue OUT;
		}
	}
}
```

## 数组

1. 静态初始化
```java
int[] arr1 = new int(){1, 2, 3, 4};
int[] arr2 = {1, 2, 3, 4};
int arr3[] = new int(){1, 2, 3, 4};
```

2. 动态初始化
```java
int[] arr_dynamic = new int[10];
```

3. 内存映像
	- 栈空间中，数组变量名对应数组头元素存储地址: `arr_dynamic`->`[I@119d7047`
	- 堆空间中，数组内的一个个元素依次存储: `[I@119d7047`开始 0, 1, 2...

4. Java中，**数组的长度是固定的**，所以不能删除元素
	- 创建新数组，跳过指定元素
	- 把删除元素用特定字符替换
	- 用列表 ArrayList
## 函数
### 重载

- 名字相同，形参不同

> 形参不同指：个数、类型、顺序

## 类

### 构造器

1. 如果没有指定构造器，会自动生成
2. 如果定义了有参数的构造器，默认构造器不会生成
3. 子类的构造器都会默认调用父类构造器，并且这是必须的

```java
public class Father{
	private Father(){}
	public Father(String name) {}
}
public class Son extends Father{
	public Son(){
		super("Java");
	}
}
```

### Public

1. 一个代码文件中，只能有一个 public 的类，并且类名和文件名相同
```java
// Object.java
public class Object {
	String name;
	int cost;
}

class Object_A{

}
```

### Static

> 类中的成员变量or成员函数，分为类变量or实例变量和类函数or实例函数
- 类变量：有 static 修饰，所有类对象共享。
- 实例变量：无 static 修饰，属于每个对象。

1. 类变量可以用：class.variable 访问(推荐)，也可以用 object.variable 访问。
2. 实例变量只能用 object.variable 访问。
3. 工具类一般不允许实例化
```java
public class Util{
	private Util{}
	public static void PostHttp(){
	}
}
```

4. 静态代码块可以在类加载的时候执行，只执行一次，
```java
public class Util(){
	public static int count;
	static {
		count =  1;
	}
}
```

>静态代码块 > 实例代码块 > 构造器

### 权限修饰符

1. private: 本类
2. 缺省: 本类、同一个 package 的类
3. protected: 本类、同一个 package 的任意类，任意 package 的
4. public: 所有类

### 继承

1. 任何类都是 Object 类的子类
2. Java 里面的类只能单继承
3. 子类可以和父类有相同的成员变量，**访问遵循就近原则**,可以通过`super.variable`访问父类成员。
#### 方法重写

1. 通过 Override 注解检查重写方法的格式
```java
public class Son extends Father{
	@Override
	public void print(){}
}
```

2. **重写方法访问权限必须大等于父类方法的权限**
3. **私有、静态方法**不能被重写

### 多态

> 编译看左边，运行看右边
> 多态指的是**成员方法**的多态，而不是成**员变量**

```java
People p1 = Teacher();
```

假如`People p1 = Teacher()`那么`p1.name`是 People 的成员变量。

| 阶段   | 看哪里 | 决定什么               |
| ---- | --- | ------------------ |
| 编译阶段 | 左边  | 能调用哪些方法（基于引用类型）    |
| 运行阶段 | 右边  | 实际调用哪个方法（基于实际对象类型） |
## Package
·
1. 同一个 package 下的类可以直接访问
2. 其他 package 下的类需要导入

```java
import com.xilyfe.pkg.util.PostHttp;
int res = PostHttp();
```

3. java.lang 下面的包不需要导入，其他需要
4. 不同 package 下如果有相同的函数/类名，第二个开始调用需要路径
```java
int res1 = PostHttp();
int res2 = com.xilyfe.pkg.http.PostHttp();
```

5. 多态下不能使用子类独有的成员方法

> 解决方法：**强制类型转换后**，使用子类对象使用成员方法

```java
People p1 = new Student();
p1.study(); // error
Student s1 = (Student)p1;
s1.study();
```

> 强制类型转换需要注意：**只有子类和父类可以相互转化**

```java
public static void execute(People p){
	if (p instanceof Student) p.study();
	else if (p instanceof Teacher) p.teach();
}

```


### Final

- final class - 不可被继承
- final function - 不可被重写
- final variable - 不可被修改

> final 修饰的变量地址不能变，但是指向的对象可以变(例如final一个class，里面的成员变量可以改变)


### 抽象类

1. 抽象类不一定有抽象方法，有抽象方法一定是抽象类
2. 抽象类不能实例化，只能继承
3. 继承了抽象类，必须实现全部抽象方法，否则他也得定义为抽象类

### 内部类

1. 成员内部类

```java
public class Outer{
	private String name = "Outer";
	private static int age = 90;
	public class Inner{
		private String name = "Inner";
		private static int age = 80;

		private void test(){
			String name = "test";
			System.out.println(name);
			System.out.println(this.name);
			System.out.println(Outer.this.name);
		}
	}
}

public static void main(){
	// 方法1
	Outer.Inner inn = new Outer.new Inner();
	// 方法2
	Outer out = new Outer();  
	Outer.Inner inn = out.new Inner();
}
```

> JDK16之后，成员内部类才能定义静态成员变量

2. 静态内部类

```java
public class Outer{
	private String name = "Outer";
	private static int age = 90;
	public class Inner{
		private String name = "Inner";
		private static int age = 80;

		private static void test(){
			String name = "test";
			System.out.println(name);
			System.out.println(this.name);
			System.out.println(Outer.this.name);
			// System.out.println(Outer.this.age);
		}
	}
}

public static void main(){
	Outer.Inner inn = new Outer.Inner();
}
```

3. 抽象内部类

```java
abstract class Base{
	public abstract void test();
}
public static void main(){
	Base a = new Base(){
		@Override
		public void test(){
		}
	}
}
```

## 接口

- 所有成员变量都是常量 - 注意大写
- 所有成员函数都是抽象函数
- 无需声明 abstract 或者 final

### 实现

- 可以实现多个接口
- 必须实现接口中的全部函数，否则自己需要定义为抽象类

```java
public class Imp implements A, B, C{}
```

> 高版本 JDK 还有其他的实现方式

1. default

虽然不能实例化 Interface 调用 default 函数，但是可以在 Interface 的实现类中调用。
```java
interface A {
	default void function(){
		//  可以有函数体
	}
}

```

2. private

用 private 修饰之后，接口实现也不能使用，主要用于 default 方法中。

```java
public interface MyInterface {

    // public default 方法
    default void sayHello() {
        greet("Hello");
    }

    default void sayHi() {
        greet("Hi");
    }

    // private 方法只能被接口内部调用
    private void greet(String message) {
        System.out.println(message);
    }
}

```

3. static

```java
interface A{
	public static void function(){
		// ...
	}
}
A.function();
```

### 函数式接口

> 函数式接口在 Java 里指的是：**只包含一个抽象方法的接口**，它是 Lambda 表达式的基础。


- **只有一个抽象方法**（可以有多个 `default` 或 `static` 方法，但抽象的只能有一个）
- 能用 **Lambda 表达式** 或 **方法引用** 来创建它的实例


```java
@FunctionalInterface
interface MyPrinter {
    void print(String msg); // 唯一抽象方法
}

public class Demo {
    public static void main(String[] args) {
        // Lambda 实现
        MyPrinter p = msg -> System.out.println(msg);
        p.print("Hello Functional Interface!");
    }
}

```
### 多继承

> 接口与函数不同，是可以多继承的

```java
interface I1{}
interface I2{}
interface I3 extends I1, I2{}
```


> 继承了父类，又实现了接口，并且父类和几口有同名的**默认方法**，默认用父类的。
> 一个类实现了多个接口，并且接口存在同名的**默认方法**，不冲突。

1. 类优先级大于接口

```java
class Parent {
    public void hello() {
        System.out.println("Hello from Parent");
    }
}

interface MyInterface {
    default void hello() {
        System.out.println("Hello from Interface");
    }
}

class Child extends Parent implements MyInterface {
    // 不需要重写 hello 方法
}

public class Main {
    public static void main(String[] args) {
        new Child().hello();  // 输出：Hello from Parent
    }
}

```


2. 多个接口的**默认方法**冲突，实现类重写即可

```java
interface a{
	default void print(){
		//...
	}
}
interface b{
	default void print(){
		//...
	}
}
class c implements a, b{
	@Override
	public void print(){
		// 可以选择调用某一个接口的默认方法：
        A.super.print();  // 或 B.super.print();
	}
}
```


## 枚举

> 枚举是一种特殊的类
> 第一行是枚举的类型
> 后面任意都行，与 class 一致

```java
public enum Sex{
	MALE, FEMALE
}

switch (sex){
	case Sex.MALE:
		break;
	case Sex.FEMALE:
		break;
}
```
---
**枚举如何实现的**

```java
public final class Sex extends Enum<Sex>{
	public static final Sex MALE = new Sex();
	public static final Sex FEMALE = new Sex();
	private Sex(){}
	public static Sex[] values(){}
	public static Sex valueof(String){}
}
```
- 将每一个枚举类型都定义为一个实例化的类变量，这样外部就不用实例化就能使用
- 禁用构造函数，只能在内部实例化
___
**抽象枚举类**

> 假如枚举类中定义了抽象方法，那么它不能被实例化

```java
public enum Animal{
	Dog(){
		@Override
		public void sing(){
			//...
		}
	}, 
	Cat(){
		@Override
		public void sing(){
			//...
		}
	}
	public abstract void sing();
}

```

## 常用API
### String

1. 与 c++ 和 python 不同，字符串不能通过索引取指定位置元素
```java
String name = "java";
System.out.println(name[0]); // error
char[] name_arr = name.toCharArray();
System.out.println(name_arr[0]); // correct
System.out.println(name.charAt(0)); // correct
```

2. 不能用 == 判断字符串是否相同，因为 String 是一个对象，== 比较的是地址
```java
String variable1 = "java";
String variable2 = "python";
System.out.println(variable1.equals(variable2));
```

3. **字符串是不可变对象**
- 对字符串进行合并操作，会在堆内存中创建新的对象，然后将变量指向它
- 用""写出的字符串对象，都会存储到堆区的字符串常量池，**相同内容**只保留一份
```java
String name_1 = "java";
char[] name_arr = {'j', 'a', 'v', 'a'};
String name_2 = new String(name_arr);
```

4. 编译优化
```java
String s1 = "ab";
String s2 = s1 + "c"; // 堆区创建2个对象
// -------------------------------------------
String s3 = "a" + "b" + "c"; // 堆区创建一个对象
```

### Arrays

```java
public static String toString(int arr[]);
public static <T>[] copyOfRange(<T>[] arr, int sta, int end); // [sta, end)
public static <T>[] copyOf(<T>[] arr, int length); // 超出补0
Arrays.setAll(arr, new IntToDoubleFunction() {
	@Override
	public double applyAsDouble(int value){
		return prices[value] * 0.8;
	}
});
public static <T>[] sort(<T>[] arr);
```


#### 自定义排序规则

1. 类实现 Comparable 接口

```java
class Student implements Comparable{
	public int age;
	public double score;
	public String name;

	@Override
	public int compareTo(Student o){
		if (score > o.score) return 1;
		else if (score < o.score) return -1;
		return 0;
	}
}
```

2. 在 sort 方法内部，创建比较器接口的匿名内部类

```java
Arrays.sort(students, new Comparator<Student>() {
	@Override
	public int compare(Student s1, Student s2){
		return 0;
	}
});
```
### Object

Java 中所有类都继承自 Object 类，可以使用它的一些方法

#### toString

- 默认情况下，`Object.toString()`返回的是类似`com.xilyfe.java_1.Student@xxxxxx`。
- 可以通过重写 Class 的 `toString`方法修改返回值。

```java
public class Student {

	@Override
	public String toString(){
		return "Student";
	}
}
```

#### equals

- `Object.equals`对比的是两个 Object 对象的地址
- 可以重写来自定义比较规则

```java
public class Student {

	@Override
	public boolean equals(const Student& stu){
		if (this == stu) return true;
		if (stu == null || getClass() != stu.getClass()) return false;
		return Objects.equals(this.name, stu.name);
	}
}

```

> `Objects.equals(a, b)`和`a.equals(b)`的区别在于，前者会对 a 先进行非空判断避免报错。

#### clone

```java
public class User implements Cloneable {

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}
}
```

> 实现 Cloneable 是一个标识，意味着这个类可以克隆

`Object.clone`默认是**潜克隆**，意味着它只是拷贝一个地址，如果变化了克隆的对象也会变。

要实现深克隆就需要对除了**基本类型**和**字符串**的其他对象之外创建新对象

```java
public class User implements Cloneable {

	@Override
	protected Object clone() throws CloneNotSupportedException {
		User u2 = (User)super.clone();
		u2.scores = us.scores.clone();
		return u2;
	}
}
```

### StringBuilder

> 对字符串进行大量操作，使用 StringBuilder 而不是 String

String 每次操作字符串都会创建新的对象，而 StringBuilder 会在原对象上进行操作，效率高非常多


StringBuilder 和 StringBuffer 区别是后者是线程安全的。


### StringJoiner

- 可以指定分隔符，前缀和后缀：`StringJoiner sj = new StringJoiner(delimiter, prefix, suffix)`

- 通过`sj.add()`添加元素


### BigDecimal

> 使用基本浮点类型操作浮点数存在局限性，没法处理类似`0.1+0.2=0.3`,对于浮点数运算逐位计算更准确


```java
double f1 = 0.1;
double f2 = 0.2;

// 方法1
BigDecimal bd1 = new BigDecimal(Double.toString(f1));
// 推荐方法
BigDecimal bd2 = BigDecimal.valueof(f1);
```

> 注意使用`new BigDecimal(String)`而不是`new BigDecimal(Double)`,后者还存在精度问题


```java
BigDecimal bd1 = BigDecimal.valueof(0.1);
BigDecimal bd2 = BigDecimal.valueof(0.2);

BigDecimal bd3 = bd1.add(bd2);
BigDecimal bd4 = bd1.multiply(bd2);
BigDecimal bd5 = bd1.divide(bd2);
```

> 注意 BigDecimal 进行除法时候如果无法整除会报错，需要指定精度和舍入方式。
> `BigDeciaml bd6 = bd1.divide(bd2, 4, RoundingMode.HALF_UP);`


## 时间管理

### Date

```java
// 获取当前日期
Date d = new Date();

// 获取毫秒数
long time = d.getTime();

// 毫米转日期
long t2 = time + 2 * 1000;
Date d2 = new Date(t2);
Date d3 = new Date();
d3.setTime(t2);

```

### SimpleDateFormate

> 这个类的作用是方便格式化输出日期


```java
Date d = new Date();
long time = d.getTime();
SimpleDateFormate sdf = new SimpleDateFormate("yyyy-mm-dd HH-mm-ss EEE a");
System.out.println(sdf.formate(d));
System.out.println(sdf.formate(time));


// 日期转Date
String dateStr = "2002-10-26 10-24-45"
SimpleDateFormate sdf2 = new SimpleDateFormate("yyyy-mm-dd HH-mm-ss");
Date d2 = sdf2.parse(dateStr);
```

### Calendar

| 方法                                   | 说明         |     |
| ------------------------------------ | ---------- | --- |
| public static Calendar getInstance() | 获得日历单例     |     |
| public int get(field)                | 获得日历某个字段信息 |     |
| public final Date getTime()          | 获得Date对象   |     |
| public long getTimeinMillis()        | 获得毫秒数      |     |
| public void set(field, value)        | 修改某个字段     |     |
| public void add(field, value)        | 增加/减少某个字段  |     |
> field 是 Calendar 提供的一个枚举类，例如 `Calendar.MONTH`


### LocalDate

- **表示**：只有日期（年、月、日），无时间信息。
- **常用方法**：

```java
import java.time.LocalDate;

LocalDate today = LocalDate.now();                // 当前日期
LocalDate birthday = LocalDate.of(2000, 5, 20);   // 指定日期
LocalDate parsed = LocalDate.parse("2025-08-08"); // 字符串解析

LocalDate tomorrow = today.plusDays(1);           // 加一天
LocalDate lastWeek = today.minusWeeks(1);         // 减一周
int year = today.getYear();                       // 获取年份
int dayOfMonth = today.getDayOfMonth();           // 获取日
```

---

### LocalTime

- **表示**：只有时间（时、分、秒、纳秒），无日期信息。
- **常用方法**：

```java
import java.time.LocalTime;

LocalTime now = LocalTime.now();                     // 当前时间
LocalTime lunch = LocalTime.of(12, 30);               // 指定时间
LocalTime parsed = LocalTime.parse("08:15:30");       // 字符串解析

LocalTime nextHour = now.plusHours(1);                // 加一小时
int hour = now.getHour();                             // 获取小时
int minute = now.getMinute();                         // 获取分钟
```

---

### LocalDateTime

- **表示**：日期 + 时间，不包含时区信息
- **常用方法**

```java
import java.time.LocalDateTime;

LocalDateTime now = LocalDateTime.now();                  // 当前日期时间
LocalDateTime meeting = LocalDateTime.of(2025, 8, 8, 14, 30); // 指定日期时间
LocalDateTime parsed = LocalDateTime.parse("2025-08-08T14:30:00");

LocalDateTime tomorrowSameTime = now.plusDays(1);         // 加一天
LocalDate datePart = now.toLocalDate();                   // 提取日期部分
LocalTime timePart = now.toLocalTime();                   // 提取时间部分
```


> 它们都是 **不可变对象**，任何加减操作都会返回新的实例。


## 包装类

> 包装类就是把基本类型数据包装成对象

```java

// 实例化
Integer a1 = Integer.valueof(1);

// 自动装箱
Integer a2 = 1;

// 自动拆箱
int a3 = a2;

// 泛型不支持基本类型，所以需要包装类
ArrayList<Integer> list = new ArrayList<Integer>();
list.add(1);
int a4 = list.get(1);
```

---

字符串和数字的类型转换

```java
Integer a = 12;
String rs1 = Integer.toString(a);
String rs2 = a.toString();
String rs3 = a + "";
```

```java
String ageStr = "12";
Integer a1 = Integer.ParseInt(ageStr);
Integer a2 = Integer.valueof(ageStr);
String scoreStr = "12.5";
Float f1 = Float.ParseFloat(scoreFloat);
Float f2 = Float.valueof(scoreFloat);

```
## 设计模式
### 单例模式

1. 懒汉式
```java
public class Singleton{
	private static instance = new Singleton();
	private Singleton(){}
	public GetInstance(){
		return instance;
	}
}
```
2. 饿汉式
```java
public class Singleton{
	private static instance;
	private Singleton(){}
	public GetInstance(){
		if (instance == null){
			instance = new Singleton();
		}
		return instance;
	}
}
```
### 模板方法

```java
public abstract class People{
	public void func(){
		// 重复任务1
		this.todo();
		// 重复任务2
	}
	public abstract void todo(){}
}
public class Student{
	@Override
	public void todo(){
		// ...
	}
}
```

## 泛型

### 泛型类

```java
public class ArrayList<E> {
	private object[] objs = new object[10];
	private int size = 0;
	private ArrayList();
	public void add(E e){
		objs[size++] = e;
	}
	public E get(int index){
		return objs[index];
	}
}

```


### 泛型方法

```java
public static <E extends Car> go(ArrayList<E> car){
	// ...
}
```

```java
public static void go(ArrayList<? extends Car> car){
	// ...
}
```

> 上限: extends 子类
> 下限: super 父类

---

1. 泛型擦除：泛型工作于编译阶段，当代码编译为 class 之后就不存在泛型了。
2. 泛型不支持 Java 的基本类型：int, float等


## Lambda

> Lambda 表达式只能简化函数式接口的匿名内部类

```java
public interface Singer{
	public void sing();
}

Singer singer = () -> {
	System.out.println("singer interface");
};

```

**简化**
1. 可以不写类型
2. 只有一个参数可以不加括号

## 方法引用

### 静态方法引用

```java
public class compareByData {
	public static int compareTo(Student o1, Student o2){
		return o1.getAge() - o2.getAge();
	}
}

// Arrays.sort(students, (o1, o2) -> compareByData.compareTo(o1, o2));
Arrays.sort(students, compareByData::compareTo);
```

> 要求是前后参数一致


### 实例方法引用

```java
public class compareByData {
	public int compareTo(Student o1, Student o2){
		return o1.getAge() - o2.getAge();
	}
}

compareByData cbd = new compareByData();
// Arrays.sort(students, (o1, o2) -> cbd.compareTo(o1, o2));
Arrays.sort(students, cbd::compareTo);
```


### 特定类型方法引用

```java
String names[] = {"a", "b", "c"};
// Arrays.sort(names, (o1, o2) -> o1.compareToIgnore(o2));
Arrays.sort(names, String::compareToIgnore);
```

> 第一个参数需要是主调
