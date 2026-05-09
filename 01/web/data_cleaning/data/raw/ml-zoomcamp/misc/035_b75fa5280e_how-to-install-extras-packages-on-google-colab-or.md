---
id: b75fa5280e
question: How to install extras packages on Google Colab or Kaggle?
sort_order: 35
---

To install extra packages on Google Colab or Kaggle, you can use the following methods:

1. **Using PIP:**
   
   Execute the following command in a cell:
   
   ```bash
   !pip install tensorflow[and-cuda]==2.14
   ```

2. **Using Conda:**
   
   You can also use Conda commands. For example:
   
   ```bash
   !conda install pandas --yes
   ```
   
   The option `--yes` allows the installation to proceed automatically when you see the "Proceed ([y]/n)?" message.