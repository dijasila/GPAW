import re
import sphinx_rtd_theme


rtd_path = sphinx_rtd_theme.get_html_theme_path()
with open(rtd_path + '/sphinx_rtd_theme/static/css/theme.css') as fd:
    css = fd.read()
        
    
def replace(match):
    s = match.group()
    i = match.start()
    t = match.string[i - 6:i]
    if t not in ['color:', 'round:']:
        print(match.string[i - 16:i + 16])
    if len(s) == 8:
        return s[0:3] + s[5:7] + s[3:5] + s[7]
    if len(s) == 5:
        return s[0:2] + s[3] + s[2] + s[4]
    1 / 0
        
new = re.sub('#[a-fA-F0-9]{3,6}[;}]', replace, css)
print(len(css), len(new))
with open('theme.css', 'w') as fd:
    fd.write(new)
