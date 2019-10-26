#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools

import sqlite3
from igraph import Graph

# Set hardcoded GRASSBIN until grass-session 0.2 is released
#if not os.environ.get('GRASSBIN'):
#    os.environ['GRASSBIN'] = '/usr/local/bin/grass79'
from grass_session import Session

"""
gscript.run_command('v.overlay', flags='t', overwrite=True, verbose=True,
                    ainput='DTM_1m_streams_cleaned@p_CulvertFragmentation_test2',
                    atype='line', binput='FKB_Hav@p_CulvertFragmentation_test2', # alayer='1',
                    operator='not', output='DTM_1m_streams_cleaned_clipped',
                    snap='0.01')
"""
occurrences_shp = '/home/stefan/seatrout/CollatedData.2019-09-03.Anadromous.25833.shp'
occurrences = 'CollatedData_2019_05_13_Anadromous'
streams = 'DTM_1m_streams_cleaned_clipped@p_CulvertFragmentation_test2'
roads = 'N50_Barriers@p_CulvertFragmentation_test'
roads_org = 'N50_roads@p_CulvertFragmentation_test'
bane_org = 'N50_Bane@p_CulvertFragmentation_test'
lines = 'lines'
culverts_nvdb = 'lines'
elevation = 'DEM_1m_carved@p_CulvertFragmentation_test2'
full_network = 'full_network'

grassdb = '/home/shared/grassdata'
location = 'ETRS_33N'
mapset = 'p_CulvertFragmentation_stefan.blumentrath_oct'

with Session(gisdb=grassdb, location=location, mapset=mapset, create_opts=""):
    import grass.script as gscript
    from grass.pygrass.vector.table import *
    gscript.run_command('v.extract', overwrite=True, verbose=True,
                        input=roads_org, output='roads_l',
                        where="MEDIUM = 'L'")
    gscript.run_command('v.extract', overwrite=True, verbose=True,
                        input=bane_org, output='bane_l',
                        where="MEDIUM = 'L'")

    #gscript.run_command('g.copy', vector='{0}@p_CulvertFragmentation_test2,{0}'.format(occurrences))
    #gscript.run_command('g.copy', vector='{0}@p_CulvertFragmentation_test2,{0}'.format(full_network))

    # Snap occurrences to network
    gscript.run_command('v.distance', overwrite=True, verbose=True,
                        from_=occurrences, from_type='point',
                        to=streams, to_type='line', output=lines, dmax=25,
                        column='to_cat', upload='cat', table=lines)

    gscript.run_command('v.distance', overwrite=True, verbose=True,
                        from_=lines, from_type='line',
                        to=occurrences, to_type='point',
                        dmax=0.01, upload='cat', column='from_cat')

    gscript.run_command('v.db.join',  map=lines, column='from_cat',
                        other_table=occurrences, other_column='cat',
                        subset_columns='efid')
    gscript.run_command('v.db.join',  map=full_network, layer='2', column='cat',
                        other_table=lines, other_column='to_cat',
                        subset_columns='efid')

    gscript.run_command('v.to.points', flags='r', overwrite=True, verbose=True,
                        input=lines, type='line,area', use='end',
                        output='{}_snaped'.format(occurrences))

    # Extract culverts fro network
    gscript.run_command('v.patch', flags='b', overwrite=True, verbose=True,
                        input='{},{}'.format(streams, roads), output='tmp')
    gscript.run_command('v.clean', flags='c', overwrite=True, verbose=True,
                        input='tmp', type='line', output='tmp2',
                        error='intersections', tool='break')
    gscript.run_command('v.category', input='intersections', type='point',
                        option='add', output='intersections_cat')

    gscript.run_command('v.extract', overwrite=True, verbose=True,
                        input='intersections_cat', output='intersection_points',
                        type='point')

    # Merge occurrences and culverts and connect them with network
    gscript.run_command('v.patch', flags='b', overwrite=True, verbose=True,
                        input='{}_snaped,intersection_points'.format(occurrences),
                        output='nodes_new')

    gscript.run_command('v.net', overwrite=True, flags='cs', input=streams,
                        points="nodes_new", output="TEST", operation="connect",
                        arc_layer="1", arc_type="line", node_layer="2",
                        threshold=0.01, turn_layer="3", turn_cat_layer="4")

    gscript.run_command('v.category', overwrite=True, verbose=True,
                        input="TEST",layer="3",
                        type="line", output="TEST2",
                        option="add", cat=1, step=1)
    gscript.run_command('v.extract', overwrite=True, input="TEST2",
                        type="line", layer="3", output="TEST3")

"""
gscript.run_command('v.category', overwrite=True, verbose=True,
                    input="TEST3@p_CulvertFragmentation_test2",layer="2",
                    type="point", output="TEST4@p_CulvertFragmentation_test2",
                    option="add", cat=1, step=1)
"""

## Read Network data from vector map
scales = [1, 5, 11, 21, 31, 51]
input = "TEST3"
node_layer = '2'
arc_layer = '3'
table = '{}_{}'.format(output, node_layer)
table_arcs = '{}_{}'.format(output, arc_layer)
with Session(gisdb=grassdb, location=location, mapset=mapset, create_opts=""):
    gscript.run_command('g.mapsets', operation='add', mapset='p_CulvertFragmentation_test2')
    gscript.run_command('v.net', flags='c', input=input, arc_layer=arc_layer,
                        output=full_network, operation='nodes',
                        node_layer=node_layer, quiet=True,
                        overwrite=True)

    gscript.run_command('v.db.addtable', verbose=True,
                        map=full_network, layer=arc_layer,
                        table='{}_3'.format(full_network))

    # Compute slope for network segments
    raster_maps = ','.join(['slope_{}'.format(scale) for scale in scales])
    gscript.run_command('g.region', raster='slope_{}'.format(scales[0]), flags='p')

    gscript.run_command('r.slope.direction',
                     elevation=elevation,
                     direction='DTM_1m_flow_dir_cleaned@p_CulvertFragmentation_test2',
                     dir_type='auto', steps='1,11,21,31', slope_measure='degree',
                     output='slope_1,slope_11,slope_21,slope_31')

    gscript.run_command('v.rast.stats', map=full_network,
                        layer=arc_layer, raster=raster_maps, column_prefix=raster_maps,
                        flags='c', method='maximum,average,stddev')
    gscript.run_command('v.to.db', map=full_network,
                        layer=arc_layer, type='line', option='length',column='length_m',
                        quiet=True)

with Session(gisdb=grassdb, location=location, mapset=mapset, create_opts=""):
    # Data has to be parsed or written to file as StringIO objects are not supported by igraph
    # https://github.com/igraph/python-igraph/issues/8
    net = gscript.read_command('v.net', input=full_network, points=full_network,
                               node_layer=node_layer, arc_layer=arc_layer,
                               operation='report', quiet=True).rstrip('\n').split('\n')

    from io import BytesIO
    import numpy as np
    from grass.script import encode
    edge_attrs = np.genfromtxt(BytesIO(gscript.read_command('v.db.select', map=full_network, layer='3', separator=',', quiet=True).rstrip('\n').encode()), dtype=None, names=True, delimiter=',')

    # Parse network data and extract vertices, edges and edge names
    edges = []
    vertices = []
    edge_cat = []
    for l in net:
        if l != '':
            # Names for edges and vertices have to be of type string
            # Names (cat) for edges
            edge_cat.append(l.split(' ')[0])

            # From- and to-vertices for edges
            edges.append((l.split(' ')[1], l.split(' ')[2]))

            # Names (cat) for from-vertices
            vertices.append(l.split(' ')[1])

            # Names (cat) for to-vertices
            vertices.append(l.split(' ')[2])

    # Create Graph object
    g = Graph().as_directed()

    # Add vertices with names
    vertices.sort()
    vertices = set(vertices)
    g.add_vertices(list(vertices))

    # Add edges with names
    g.add_edges(edges)
    g.es['cat'] = edge_cat

    # Add edge attributes
    for colname in edge_attrs.dtype.names[1:]:
        g.es[colname] = edge_attrs[colname]

    # Mark occurrence vertices
    occs = gscript.read_command('v.distance', overwrite=True, verbose=True,
                                from_=lines, from_type='line', to_layer=node_layer,
                                to=full_network, to_type='point', flags='p', dmax=0.01,
                                upload='cat').split('\n')[1:-1]

    occ_cats = [occ.split('|')[1] for occ in occs]

    g.vs['occurrence'] = 0
    g.vs.select(name_in=occ_cats)['occurrence'] = 1

    # Mark culvert vertices
    culverts = gscript.read_command('v.distance', overwrite=True, verbose=True,
                                    from_='intersection_points', from_type='point',
                                    to_layer=node_layer, to=full_network, to_type='point',
                                    flags='p', dmax=0.1, upload='cat'
                                    ).split('\n')[1:-1]

    culvert_cats = [culvs.split('|')[1] for culvs in culverts if not 'null' in culvs]

    g.vs['culvert'] = 0
    g.vs.select(name_in=set(culvert_cats))['culvert'] = 1

    # Mark bridge vertices
    bridges_r = gscript.read_command('v.distance', overwrite=True, verbose=True,
                                    from_='roads_l', from_type='line',
                                    to_layer=node_layer, to=full_network, to_type='point',
                                    flags='p', dmax=5, upload='cat'
                                    ).split('\n')[1:-1]
    bridges_b = gscript.read_command('v.distance', overwrite=True, verbose=True,
                                    from_='bane_l', from_type='line',
                                    to_layer=node_layer, to=full_network, to_type='point',
                                    flags='p', dmax=5, upload='cat'
                                    ).split('\n')[1:-1]

    bridge_cats = [bridge.split('|')[1] for bridge in bridges_r if not 'null' in bridge] + \
                  [bridge.split('|')[1] for bridge in bridges_b if not 'null' in bridge]

    g.vs['bridge'] = 0
    g.vs.select(name_in=set(bridge_cats))['bridge'] = 1
    g.vs.select(bridge=1)['culvert'] = 0

    # Identify clusters
    clusters = g.as_undirected().clusters()
    # max(clusters.sizes())
    g.vs['cl'] = g.as_undirected().clusters().membership

    # Remove clusters without occurrences
    clusters_rel = set(g.vs.select(occurrence=1)['cl'])

    g.delete_vertices(g.vs.select(cl_notin=clusters_rel))

    # Compute number of vertices that can be reached from each vertex
    # Indicates upstream or downstream position of a node
    g.vs['nbh'] = g.neighborhood_size(mode='out', order=g.diameter())

    # Compute incoming degree centrality
    # sources have incoming degree centrality of 0
    g.vs['indegree'] = g.degree(mode="in")

    # Compute outgoing degree centrality
    # outlets have outgoing degree centrality of 0
    g.vs['outdegree'] = g.degree(mode="out")

    g.vs['uddegree'] = g.degree(mode="all")

    g.vs['node_type'] = None # 'other'
    g.vs.select(indegree=0)['node_type'] = 'source'
    g.vs.select(outdegree=0)['node_type'] = 'outlet'
    g.vs.select(indegree_gt=1, outdegree_gt=0)['node_type'] = 'confluence'
    g.vs.select(indegree=1, outdegree=1)['node_type'] = 'other'

    # Compute upstream network from culvert
    #g.neighborhood(g.vs.select(occurrence=1), mode='IN', order=g.diameter())
    rel = set(itertools.chain(*g.neighborhood(g.vs.select(occurrence=1), mode='OUT', order=g.diameter())))
    culvs = g.vs[rel].select(culvert_eq=1, )
    upstream_nbh = np.array(g.neighborhood(culvs, mode='IN', order=g.diameter()))
    downstream_nbh = np.array(g.neighborhood(culvs, mode='OUT', order=g.diameter()))
    g.vs['upstream_length_m'] = 0
    g.vs['upstream_culverts_n'] = 0
    g.vs['upstream_length_m_not_uniq'] = 0
    g.vs['upstream_length_m_uniq'] = 0

    # The following should be parallelized due to slow inner loop!
    for culv_idx, nbh in enumerate(upstream_nbh):
        #sum(g.es[set(itertools.chain(*g.get_shortest_paths(culv, to=g.vs.select(node_type='source'), mode='IN', output='epath')))]['length_m'])
        upstream_graph = g.subgraph(set(nbh))
        name = culvs[culv_idx]['name']
        g.vs.select(name_eq=name)['upstream_length_m'] = sum(upstream_graph.es['length_m'])
        g.vs.select(name_eq=name)['upstream_culverts_n'] = sum(upstream_graph.vs['culvert']) - 1
        if sum(upstream_graph.vs['culvert']) - 1 > 0:
            upstream_graph_not_uniq = g.subgraph(np.unique(np.hstack(upstream_graph.neighborhood(upstream_graph.vs.select(name_ne=name, culvert_eq=1), mode='IN', order=g.diameter()))))
            g.vs.select(name_eq=name)['upstream_length_m_not_uniq'] = sum(upstream_graph_not_uniq.es['length_m'])
            g.vs.select(name_eq=name)['upstream_length_m_uniq'] = g.vs.select(name_eq=name)['upstream_length_m'][0] - g.vs.select(name_eq=name)['upstream_length_m_not_uniq'][0]

    for culv_idx, nbh in enumerate(downstream_nbh):
        downstream_graph = g.subgraph(set(nbh))
        name = culvs[culv_idx]['name']
        g.vs.select(name_eq=name)['downstream_culverts_n'] = sum(downstream_graph.vs['culvert']) - 1

    # rel = set(itertools.chain(*[itertools.chain(*g.neighborhood(g.vs.select(occurrence=1), mode='IN', order=g.diameter())),
    #                             itertools.chain(*g.neighborhood(g.vs.select(occurrence=1), mode='OUT', order=g.diameter()))]))

    g = g.subgraph(rel)

    anadrome_edges = g.es['cat']

    def lwa(val, length):
        return sum(val[~np.isnan(val)] * length[~np.isnan(val)]) / sum(length[~np.isnan(val)])

    import itertools
    # Loop over occurrences
    measures = ['maximum', 'average', 'stddev']
    for v in g.vs.select(occurrence=1):
        outlet = g.vs.select(node_type='outlet', cl=v['cl'])
        edges_downstreams = g.get_shortest_paths(v, to=outlet, mode='OUT', output='epath')
        downstream_edges = g.es[itertools.chain.from_iterable(edges_downstreams)]
        vertices_downstreams = g.get_shortest_paths(v, to=outlet, mode='OUT', output='vpath')
        downstreams_vertices = g.vs[itertools.chain.from_iterable(vertices_downstreams)]
        #g.vs[v.index]['outlet']


        if len(downstream_edges) > 0:
            length_weight = np.array(downstream_edges['length_m'])

            for measure in itertools.product(scales, measures):
                measure_name = 'slope_{}_{}'.format(*measure)
                measure_value = np.array(downstream_edges[measure_name])

                if measure[1] == 'maximum':
                    measure_agg = max(measure_value[~np.isnan(measure_value)])
                else:
                    measure_agg = lwa(measure_value, length_weight)

                g.vs[v.index]['downstream_{}'.format(measure_name)] = measure_agg

        if len(downstreams_vertices) >= 0:
            g.vs[v.index]['culverts_downstream_n'] = sum(downstreams_vertices['culvert'])
            g.vs[v.index]['culverts_downstream'] = ','.join(downstreams_vertices.select(culvert=1)['name'])

    # Loop over culverts
    for v in g.vs.select(culvert=1):
        occ = g.vs.select(occurrence=1, cl=v['cl'])
        edges_upstreams = g.get_shortest_paths(v, to=occ, mode='IN', output='epath')
        vertices_upstreams = g.get_shortest_paths(v, to=occ, mode='IN', output='vpath')
        upstream_edges = g.es[itertools.chain.from_iterable(edges_upstreams)]
        upstream_vertices = g.vs[itertools.chain.from_iterable(vertices_upstreams)]
        if len(upstream_vertices) >= 0:
            g.vs[v.index]['occurrences_upstream_n'] = sum(downstreams_vertices['occurrence'])
            g.vs[v.index]['occurrences_upstream'] = ','.join(upstream_vertices.select(occurrence=1)['name'])



    g.vs['betweenness'] = g.betweenness(directed=False)

    """
    line_attrs = []
    for edge in g.es:
        line_attrs.append([min(g.vs[edge.source]['nbh'], g.vs[edge.target]['nbh']),
                           int(edge['cat'])])
    # Compute core edges
    #g.vs.select(outdegree_le=1, indegree_le=1] #== 1 or  g.degree(type="out")
    """

    gscript.verbose(_("Writing result to table..."))

    def item_check(item):
        if item:
            if isinstance(item, int):
                item_type = 'int'
            elif isinstance(item, float):
                item_type = 'float'
            else:
                item_type = 'other'
        else:
            item_type = 'None'
        return item_type

    # Write results back to attribute table
    # Note: Backend depenent! For a more general solution this has to be handled
    path = '$GISDBASE/$LOCATION_NAME/$MAPSET/sqlite/sqlite.db'
    conn = sqlite3.connect(get_path(path))
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS {}'.format(table))

    create_str = 'CREATE TABLE {}('.format(table)

    cols = []
    col_types = {}
    for attr in g.vs.attributes():
        if attr == 'name':
            col_name = 'cat'
            col_type = 'integer'
            map_function = int
        else:
            col_name = attr
            ltypes = set([item_check(attr_val) for attr_val in g.vs[attr]])
            if 'other' in ltypes:
                col_type = 'text'
                map_function = str
            elif 'float' in ltypes:
                col_type = 'real'
                map_function = float
            else:
                col_type = 'integer'
                map_function = int

        col_types[attr] = map_function
        cols.append('{} {}'.format(col_name, col_type))

    create_str += ','.join(cols)
    create_str += ')'

    # Create temporary table
    c.execute(create_str)
    conn.commit()

    # Get Attributes
    attrs = []
    for n in g.vs:
        attr_list = [col_types[attr](n[attr]) if n[attr] else None for attr in g.vs.attributes()]
        attrs.append(tuple(attr_list))


    # Insert data into temporary table
    c.executemany('INSERT INTO {} VALUES ({})'.format(table,
                                                      ','.join(['?'] * len(g.vs.attributes()))), attrs)

    # Save (commit) the changes
    conn.commit()

    # Mark edges downstreams from occurrences
    gscript.run_command('v.db.addcolumn', map=full_network, layer=arc_layer,
                        column='anadrome integer')
    c.executemany('UPDATE {} SET anadrome = ? WHERE cat = ?'.format(table_arcs), [(1, int(x)) for x in anadrome_edges])

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()

    # Connect table to output node layer
    gscript.run_command('v.db.connect', map=full_network, table=table, layer=node_layer, flags='o')

    # Join temporary table to output
    #gscript.run_command('v.db.join', map=full_network, layer=node_layer,
    #                    column='cat', other_table=tmpTable,
    #                    other_column='cat', quiet=True)

with Session(gisdb=grassdb, location=location, mapset=mapset, create_opts=""):
    #gscript.run_command('v.db.addtable', map=full_network, layer=2, key='cat')
    #gscript.run_command('v.to.db', map=full_network, layer=node_layer, option='cat', columns='cat', quiet=True)
    gscript.run_command('g.copy', vector='N50_roads@p_CulvertFragmentation_test,N50_roads')
    gscript.run_command('g.copy', vector='N50_Bane@p_CulvertFragmentation_test,N50_Bane')
    gscript.run_command('v.db.addcolumn', overwrite=True, verbose=True,
                        map=full_network, layer='2', column='N50_cat integer')

    for map in ['N50_roads', 'N50_Bane']:
        gscript.run_command('v.distance', overwrite=True, verbose=True,
                            from_=full_network, from_type='point', from_layer='2',
                            to=map, to_type='line',
                            dmax=0.01, upload='cat', column='N50_cat')

        if 'Bane' not in map:
            columns = 'MEDIUM,OBJTYPE,VEGKATEGORI'
        else:
            columns = 'MEDIUM,OBJTYPE,SPORANTALL'
        gscript.run_command('v.db.join', map=full_network, layer='2', column='N50_cat',
                            other_table=map, other_column='cat',
                            subset_columns=columns)

    columns = {
               60: 'Brukategori,Materialtype,Byggverkstype,Lengste_spenn',
               78: 'Type',
               79: 'Materialtype,Type_innlC8p,Type_utlC8p,Lengde,GjennomlC8p_for_elv_bekk,Diameter__innvendig,Tverrsnittsform,Spesielle_hensyn_fiskevandring',
               }

    for nvdbid in columns.keys():
        #gscript.run_command('g.copy', vector='NVDB_object_{0}@p_CulvertFragmentation_stefan.blumentrath,NVDB_object_{0}'.format(nvdbid))
        map = 'NVDB_object_{}'.format(nvdbid)
        gscript.run_command('g.copy', vector='{0}@p_CulvertFragmentation_test2,{0}'.format(map))

        gscript.run_command('v.db.addcolumn', overwrite=True, verbose=True,
                            map=full_network, layer='2', column='NVDB_{}_cat integer'.format(nvdbid))
        gscript.run_command('v.distance', overwrite=True, verbose=True,
                            from_=full_network, from_type='point', from_layer='2',
                            to=map, to_type='line,point',
                            dmax=10, upload='cat', column='NVDB_{}_cat'.format(nvdbid))
        gscript.run_command('v.db.join', map=full_network, layer='2', column='NVDB_{}_cat'.format(nvdbid),
                            other_table=map, other_column='cat',
                            subset_columns=columns[nvdbid])

    gscript.run_command('v.to.rast', overwrite=True, verbose=True,
                        input=full_network, layer='2',
                        type='point', where="node_type = 'outlet'", output='outlets',
                        use='cat', memory=30000)
    gscript.run_command('r.watershed', flags='ma', overwrite=True, verbose=True,
                        elevation=elevation,
                        accumulation='DTM_1m_streams_cleaned_flow_accum',
                        spi='DTM_1m_streams_cleaned_SPI', memory=80000)


    gscript.run_command('r.stream.distance', flags='o', overwrite=True, verbose=True,
                        stream_rast="outlets@p_CulvertFragmentation_test2",
                        direction='DTM_1m_flow_dir_cleaned@p_CulvertFragmentation_test2',
                        elevation=elevation,
                        distance='DTM_1m_streams_cleaned_distance', memory=100000)

    raster_maps = [(elevation, 'altitude'),
                   ('DTM_1m_streams_cleaned_distance', 'outlet_distance'),
                   ('DTM_1m_streams_cleaned_SPI', 'stream_power_index'),
                   ('DTM_1m_streams_cleaned_flow_accum', 'flow_accum')]

    for scale in scales:
        raster_maps.append(('slope_{}'.format(scale), 'slope_{}'.format(scale)))

    for raster in raster_maps:
        gscript.run_command('v.what.rast', layer='2', raster=raster[0],
                            column=raster[1], where="occurrence = 1 OR culvert = 1",
                            map=full_network)

with Session(gisdb=grassdb, location=location, mapset=mapset, create_opts=""):

    geopackage = '/home/stefan/seatrout_net_{}.gpkg'.format(date.today().strftime("%Y_%m_%d"))

    gscript.run_command('v.out.ogr', overwrite=True, verbose=True,
                        input=full_network, output=geopackage, format='GPKG',
                        output_layer='nodes', type='point', layer='2')

    gscript.run_command('v.out.ogr', flags='u', overwrite=True, verbose=True,
                        input=full_network, output=geopackage, format='GPKG',
                        output_layer='network', type='line', layer='3')

    for nvdbid in columns.keys():
        map = 'NVDB_object_{}'.format(nvdbid)
        gscript.run_command('v.out.ogr', flags='u', overwrite=True, verbose=True,
                            input=map, output=geopackage, format='GPKG',
                            output_layer=map)

    for map in ['N50_roads', 'N50_Bane']:
        gscript.run_command('v.out.ogr', flags='u', overwrite=True, verbose=True,
                            input=map, output=geopackage, format='GPKG',
                            output_layer=map)
