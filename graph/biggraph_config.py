
def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path="/home/shu/booking-challenge-2021/data/interim/biggraph_data/",
        edge_paths=['/home/shu/booking-challenge-2021/data/interim/relation_total/'],
        checkpoint_path='/home/shu/booking-challenge-2021/data/interim/biggraph_model/',

        # Graph structure
        entities={
            'all': {'num_partitions': 1},
        },
        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'complex_diagonal',
        }],
        dynamic_relations=True,

        # Scoring model
        dimension=400,
        global_emb=False,
        comparator='dot',

        # Training
        num_epochs=50,
        num_uniform_negs=1000,
        loss_fn='softmax',
        lr=0.1,

        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config

