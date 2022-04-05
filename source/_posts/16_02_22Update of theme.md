---
title: Update of theme
top: false
cover: false
toc: true
mathjax: true
date: 2022-02-16 22:41:29
password:
summary:
tags:
- blog building
categories:
- programming
---

# Update of theme

Yesterday, I changed my blog from the jekyll version into hexo version. A new theme suddenly got me. Thus, I decide to change my theme from plane to the volume, from 2d to 3d. Here is my records for the process:

## Install Node.js

Before install

``` cmd
npm ls -g --depth=0   //check if there is a earlier relesaed node installed
```
Then download from https://nodejs.org/en/ for a stable version

downgrade the npm into certain version (if you installed it before):

``` cmd
npm install npm@6.14.14 -g
```

Check the version:

``` cmd
C:\Users\{yourusername}>node -v
v12.0.0
C:\Users\{yourusername}>npm -v
6.9.0
```

You may want to restart at last.

**Tips**: At first I installed the latest version, but eventually find it do not support my ".js" files. After several trails, here is the stable version.

## Version control

Here is a useful distributed revision control tool: Git

https://git-scm.com/download/win

For the last install step, choose `Use Git from the Windows Command Prompt`

## New a repository

New a initialized (with readme.md) repository in your Github library.

## Install Hexo

Create a new folder in a suitable place to store your own blog files. For example, my blog files are stored in`D:\blog`。

Right click in this directory `Git Bash Here`，Open the console window of git, all our later operations will be performed in the git console。

Find this directory and enter `npm i hexo-cli -g`

You can type down`hexo -v ` to verify if the installation was successful.

Then we need to initialize our website, type down`hexo init`to initialize the folder, and then type down`npm install` to install the necessary components.

Until now, the local website configuration is also done.

Type down `hexo g` to generate a static webpage, then type down `hexo s` to open the local server.

Finally open the browser [http://localhost:4000/](https://link. zhihu.com/?target=http%3A//localhost%3A4000/), you can see your blog

You can press ctrl+c to shut down the local server.

## Connect Github with local

This is a important step which can help you deploy your blog to the cloud (I got stuck in this step for a while night).

First right-click to open git bash, and then enter the following command:

`git config --global user.name "{yourusername}"
git config --global user.email "{youremailaddress}"`

The username and email address should be modified according to the information you registered with github.

Then generate the key SSH key:

`ssh-keygen -t rsa -C "{youremailaddress}"`

Open github, click settings under the avatar, then click SSH and GPG keys to create a new SSH with any name.

Enter in git bash`cat ~/.ssh/id_rsa.pub`. Copy the output to the box and click OK to save.

Enter `ssh -T git@github.com`, if your username appears as shown in the figure below, congratulations, you succeed.

Open the _config.yml file in the root directory of the blog, which is the configuration file of the blog, where you can modify various information related to the blog.

Modify the configuration on the last line:

```bash
deploy:
  type: git
  repository: https://github.com/{yourusername}/{yourusername}.github.io.git
  branch: master
```
**Tips:** Remember to set the branch master as default or you can change the master into main in above code. Because the rule is changed after Nov, 2020: the master branch has been changed into main.  It may be used to avoid the term: "master slave".

## write articles, publish articles

First, right-click in the blog root directory to open git bash and install an extension 

`npm i hexo-deployer-git`

Then enter `hexo new post "article title"` to create a new article.

Then open the `D:\blog\source\_posts` directory, you can find that there is an additional folder and a .md file below, one is used to store your pictures and other data, and the other is your article file.

After writing the markdown file, enter `hexo g` in the root directory to generate a static web page, then enter `hexo s` to preview the effect locally, and finally enter `hexo d` to upload it to Github. 

You can see the published article in the **github.io** now! (It may need 5 to 10 minutes to deploy depending on the online server)