---
id: f580e98fdc
question: 'Installing waitress on Windows via GitBash: “waitress-serve” command not
  found'
sort_order: 36
---

Running `pip install waitress` as a command on GitBash may not download the executable file `waitress-serve.exe`. You need this file to use the commands with waitress in Git Bash. To resolve this issue:

1. Open a Jupyter notebook and run the command `pip install waitress`. This will download the executable file. You may see the following warning:
   
   ```bash
   WARNING: The script waitress-serve.exe is installed in 'c:\Users\....\anaconda3\Scripts' which is not on PATH. Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
   ```

2. Add the path where `waitress-serve.exe` is installed to GitBash's PATH:

   - Enter the following command in GitBash: `nano ~/.bashrc`
   
   - Add the path to `waitress-serve.exe` to PATH using the command:
     
     ```bash
     export PATH="/path/to/waitress:$PATH"
     ```

3. Close GitBash and open it again. You should now be able to run `waitress-serve` successfully.

