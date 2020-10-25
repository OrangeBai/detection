import wget
import config

print('Beginning file download with wget module')

url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
wget.download(url, '/Users/scott/Downloads/cat4.jpg')