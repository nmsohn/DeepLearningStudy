# Deep Learning Study

Week 1: 2020.07.27 - 2020.08.02 (Chapter 1) </br>
Week 2: 2020.08.03 - 2020.08.09 (Chapter 2) </br>
Week 3: 2020.08.17 - 2020.08.23 (Chapter 3) </br>
Week 4: 2020.08.24 - 2020.08.30 (Chapter 4)

---
## 🎉Before you start
- 한 주에 한 챕터를 요약해서 PR로 올리기
- 구현한 것이 있다면 따로 repo의 링크를 첨부하거나 kaggle, colab 링크 첨부하기
- 꼭 브랜치를 생성 후 PR 만들기

### 브랜치 만드는 법
```
git checkout master
git fetch
git pull
git checkout -b "study/ch1_<<username>>"
```

### 파일 작성 후 PR 만드는 법
```
git add .
git commit -m "<<커밋 내용>>"
git push --set-uptream origin feature/ch1_<<username>>
```
한 후 깃헙 repo로 가면 상단에 PR 버튼이 뜸.

---

### Naming Convention
#### ✏글작성시
- 파일은 마크다운으로 작성하고 파일이름은 `ch1_유저네임.md`로 하여 생성
- 챕터 폴더 내 README.md 에 파일 링크를 목차에 포함시키기
- 이미지를 첨부하는 경우 챕터 폴더 안 `image` 폴더에 넣어주기
- 이미지 이름은 `ch1_이미지설명.md`로

#### 😀PR 시
- branch 생성 후 master로 PR을 만들기
- 브랜치 이름은 `study/ch1_유저네임`으로 만들고 PR 제목에 챕터 이름을 써주기
