# EVRAZ AI CHALLENGE
## Трек 1 "Продуйте металл через Data Science"
## Решение команды "Hot-rolled-team"
> Артем Проничев
> 
> Олег Сидоршин
> 
> Даниил Кузнецов
> 
> Данил Картушов
> 
> Виктор Журавлев

## Краткое описание задачи
Предметная область: черная металлургия (продувка чугуна - ключевой процесс производства стали).

Целью хакатона является разработка модели, предсказывающей содержание углерода и температуру
чугуна во время процесса продувки металла.

При производстве стали чугун "продувается" кислородом для удаления примесей. Этот процесс идёт в среднем 15-25
минут при температуре около 1600 градусов. За процессом следит машинист дистрибутора, который на основе своего
опыта и специальных знаний определяет момент, когда процесс продувки нужно остановить. В процессе продувки
металл насыщается кислородом, а его температура увеличивается. Если "передуть" чугун – сгорит больше металла и
на выходе будет меньше стали, что приведет к потере прибыли, если "недодуть", то марка стали не будет удовлетворять
заданным критериям и нужно будет "додувать", что замедляет производительность цеха.

Разработка алгоритма прогнозирования параметров чугуна может стать отличным помощником для
машиниста и существенно улучшить производство ЕВРАЗа

Подробнее о задаче можно посмотреть <a href="https://russianhackers.notion.site/1-Data-Science-4cc89ba42de1429bbac316f59bf07a3b">здесь</a>

## Краткое описание решения

В решении задачи основной упор делался на feature engineering. 

В части моделирования использовались LightAutoML и LightGBM+Optuna.
## Результат
10 место по приватному скору из 65 команд

![10/65](img/Leader_Board.png)
