""" В этом файле содержатся все методы выбора элента из списков"""

selected_lot = None
selected_project = None
selected_discips = None
selected_actor = None
selected_winner = None
selected_currency = None

def on_lot_selected(item, storage):
    storage['selected_lot'] = item.text()
    print(f'Выбранный лот: {storage["selected_lot"]}')
    return storage

def on_project_selected(item, storage):
    storage['selected_project'] = item.text()
    print(f'Выбранный проект {storage["selected_project"]}')
    return storage

def on_discips_selected(item, storage):
    storage['selected_discipline'] = item.text()
    print(f'Выбранный Исполнитель {storage["selected_discipline"]}')
    return storage

def on_actors_selected(item, storage):
    storage['selected_actor'] = item.text()
    print(f'Выбранный Исполнитель {storage["selected_actor"]}')
    return storage

def on_winner_selected(item, storage):
    storage['selected_winner'] = item.text()
    print(f'Выбранный Поставщик {storage["selected_winner"]}')
    return storage

def on_currency_selected(item, storage):
    storage['selected_currency'] = item.text()
    print(f'Выбранная Валюта {storage["selected_currency"]}')
    return storage