作者：XuXiran YingJiahe

如何配环境问小应。他正在努力做一个可以直接exe运行而不用装包的版本
首先，将原来的五线谱放在文件目录外。（input.pdf）

![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/993b270f-4fa3-4681-92ec-57ca08be6d4c)

然后，运行score_recognition_v4文件中的main.py

调性识别一般不是很准确，但是可以手工改正。

![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/d3942552-a038-47eb-9981-65222f30ac94)


![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/17f00619-efe9-43e5-9e5d-0572b9f34854)

如果和想要的调性不一样，就直接输入即可，例如：

![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/2517bfd2-b1fd-4660-8a77-8214438c03d3)



谱号识别稍微好一点，但是也可能错误。认真核实。

![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/2f46b10b-5a36-4491-bca8-07fd4832cf35)

8是高音谱号，9是低音谱号。如果需要修改也是一样的。

![image](https://github.com/xuxiran/translate-staff-to-simple-musical-notation/assets/48015859/d4e6ba36-c94e-437f-9a15-685101cd5bfc)


最终会在input所在路径下生成output和output2.前者是直接在五线谱下写简谱。后者删除五线谱。

# translate-staff-to-simple-musical-notation
Polyphony, optical music recognition, template matching, staff, jianpu, simple-musical-notation
the "input.pdf" is the input and you will get two output.pdf as output.
We only want to translate the staff to simple-musical-notation
