package count.model;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import count.ds.AnnotatedTable;
import count.ds.Phylogeny;
import count.io.NewickParser;
import count.io.RateVariationParser;
import count.io.TableParser;

/**
 * Java reference driver: load a Count session XML (optionally gzipped),
 * compute the corrected log-likelihood and its analytical gradient for each
 * table in the session, and print the results.
 *
 * Usage:
 *   java -cp build/classes count.model.CountVerifyXML \
 *        validation/Williams2017.countxml.gz \
 *        [sessionId]               # default = first session with a model
 *        [tableName]               # default = first table in that session
 *        [minCopies]               # default = 1
 *
 * Prints, on stdout:
 *   tree=<num_leaves>/<num_nodes>
 *   table=<rows>x<cols>
 *   LL_raw=<float>
 *   LL_corrected_minK=<float>
 *   LL_empty=<float>
 *   LL_singleton=<float>
 *   GRADIENT_SURVIVAL  (one line per "node param value")
 */
public class CountVerifyXML {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: CountVerifyXML <countxml(.gz)> [sessionId] [tableName] [minCopies]");
            System.exit(2);
        }
        String path = args[0];
        String wantSession = args.length > 1 && !args[1].isEmpty() ? args[1] : null;
        String wantTable = args.length > 2 && !args[2].isEmpty() ? args[2] : null;
        int minCopies = args.length > 3 ? Integer.parseInt(args[3]) : 1;

        // ---- Load XML ----------------------------------------------------
        DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
        org.w3c.dom.Document doc;
        try (InputStream in = openMaybeGzipped(path)) {
            doc = db.parse(in);
        }

        Element session = pickSession(doc.getDocumentElement(), wantSession);
        if (session == null) {
            throw new RuntimeException("No session found (with a <model>) in " + path);
        }
        String sid = session.getAttribute("id");
        System.out.println("# session=" + sid);

        // Tree: first <tree> directly under the session.
        Element treeEl = firstChildElement(session, "tree");
        if (treeEl == null) throw new RuntimeException("session has no <tree>");
        String newick = treeEl.getTextContent();
        Phylogeny tree = NewickParser.readTree(new StringReader(newick));
        int n = tree.getNumNodes();
        System.out.println("# tree=" + tree.getNumLeaves() + "/" + n);

        // Rate model: first <model> directly under the tree.
        Element modelEl = firstChildElement(treeEl, "model");
        if (modelEl == null) throw new RuntimeException("session has no <model> under <tree>");
        BufferedReader modelReader = new BufferedReader(new StringReader(modelEl.getTextContent()));
        TreeWithRates rates = RateVariationParser.readModel(modelReader, tree).getBaseModel();
        System.out.println("# rates loaded");

        // Table: <table> children of the session.
        Element tableEl = pickTable(session, wantTable);
        if (tableEl == null) throw new RuntimeException("session has no usable <table>");
        String[] taxa = tree.getLeafNames();
        BufferedReader tableReader = new BufferedReader(new StringReader(tableEl.getTextContent()));
        AnnotatedTable table = TableParser.readTable(taxa, tableReader, false);
        System.out.println("# table=" + tableEl.getAttribute("name")
            + " " + table.getFamilyCount() + "x" + taxa.length);

        // ---- Likelihood + gradient --------------------------------------
        Likelihood lik = new Likelihood(rates, table);
        // Time just the forward LL pass (no gradient). Reset cache between runs.
        long t_warm0 = System.nanoTime();
        @SuppressWarnings("unused") double _w = lik.getLL();
        long t_warm = System.nanoTime() - t_warm0;
        System.out.printf("# java LL warmup: %.3f ms%n", t_warm / 1.0e6);
        Likelihood likTimed = new Likelihood(rates, table);
        long t0_ll = System.nanoTime();
        @SuppressWarnings("unused") double _l = likTimed.getLL();
        long t_ll = System.nanoTime() - t0_ll;
        System.out.printf("# java LL timed:  %.3f ms%n", t_ll / 1.0e6);
        double LL = lik.getLL();
        double LL_corr;
        if (minCopies == 0) LL_corr = LL;
        else if (minCopies == 1) LL_corr = lik.getCorrectedLL();
        else {
            // min_copies=2: corrected for empty + singleton
            double L0 = lik.getEmptyLL();
            double L1 = lik.getSingletonLL();
            double Lunobs = count.matek.Logarithms.add(L0, L1);
            int F = table.getFamilyCount();
            double p_obs = -Math.expm1(Lunobs);
            LL_corr = LL - F * Math.log(p_obs);
        }
        double L0 = lik.getEmptyLL();
        double L1 = lik.getSingletonLL();

        PrintStream out = System.out;
        out.println("LL_raw=" + LL);
        out.println("LL_corrected_min" + minCopies + "=" + LL_corr);
        out.println("LL_empty=" + L0);
        out.println("LL_singleton=" + L1);

        // Per-family LL for ALL families — used by the Python comparison to
        // locate families with the worst numerical disagreement.
        out.println("PER_FAMILY_LL");
        for (int f = 0; f < table.getFamilyCount(); f++) {
            Likelihood.Profile pp = lik.getProfileLikelihood(f);
            out.printf("%d %.15e%n", f, pp.getLogLikelihood());
        }

        // Dump the survival parameters for direct comparison with Python's
        // compute_survival_params output.
        out.println("SURVIVAL_PARAMS");
        for (int v = 0; v < n; v++) {
            out.printf("%d ptilde %.15e%n", v, lik.getLossParameter(v));
            out.printf("%d qtilde %.15e%n", v, lik.getDuplicationParameter(v));
            out.printf("%d gain   %.15e%n", v, lik.getGainParameter(v));
            out.printf("%d eps    %.15e%n", v, lik.getExtinction(v));
        }

        Gradient grad = new Gradient(rates, table);
        grad.setMinimumObservedCopies(minCopies);
        // warmup + timed gradient
        long t_gw0 = System.nanoTime();
        @SuppressWarnings("unused") double[] _gw = grad.getCorrectedGradient();
        long t_gw = System.nanoTime() - t_gw0;
        System.out.printf("# java grad warmup: %.3f ms%n", t_gw / 1.0e6);
        Gradient gradTimed = new Gradient(rates, table);
        gradTimed.setMinimumObservedCopies(minCopies);
        long t_g0 = System.nanoTime();
        @SuppressWarnings("unused") double[] _gt = gradTimed.getCorrectedGradient();
        long t_g = System.nanoTime() - t_g0;
        System.out.printf("# java grad timed:  %.3f ms%n", t_g / 1.0e6);
        double[] g = grad.getCorrectedGradient();

        out.println("GRADIENT_SURVIVAL");
        // index order in Java: 3*v + {GAIN=0, LOSS=1, DUP=2}
        for (int v = 0; v < n; v++) {
            out.printf("%d gain %.15e%n", v, g[3 * v + 0]);
            out.printf("%d loss %.15e%n", v, g[3 * v + 1]);
            out.printf("%d dup  %.15e%n", v, g[3 * v + 2]);
        }
        out.println("# done");
    }

    // ---- XML helpers -----------------------------------------------------

    private static InputStream openMaybeGzipped(String path) throws Exception {
        InputStream raw = new FileInputStream(path);
        return path.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
    }

    private static Element firstChildElement(Element parent, String tag) {
        NodeList kids = parent.getChildNodes();
        for (int i = 0; i < kids.getLength(); i++) {
            Node k = kids.item(i);
            if (k.getNodeType() == Node.ELEMENT_NODE && k.getNodeName().equals(tag)) {
                return (Element) k;
            }
        }
        return null;
    }

    private static List<Element> childElements(Element parent, String tag) {
        List<Element> out = new ArrayList<>();
        NodeList kids = parent.getChildNodes();
        for (int i = 0; i < kids.getLength(); i++) {
            Node k = kids.item(i);
            if (k.getNodeType() == Node.ELEMENT_NODE && k.getNodeName().equals(tag)) {
                out.add((Element) k);
            }
        }
        return out;
    }

    /**
     * Pick a session: by id if specified, else the first one that has a
     * <tree>/<model> (so we ignore "no rates fit yet" sessions).
     */
    private static Element pickSession(Element root, String wantId) {
        NodeList sessions = root.getElementsByTagName("session");
        for (int i = 0; i < sessions.getLength(); i++) {
            Element s = (Element) sessions.item(i);
            if (wantId != null) {
                if (s.getAttribute("id").equals(wantId)) return s;
            } else {
                Element t = firstChildElement(s, "tree");
                if (t != null && firstChildElement(t, "model") != null) return s;
            }
        }
        return null;
    }

    private static Element pickTable(Element session, String wantName) {
        for (Element t : childElements(session, "table")) {
            if (wantName == null) return t;
            String n = t.getAttribute("name");
            String id = t.getAttribute("id");
            if (wantName.equals(n) || wantName.equals(id)) return t;
        }
        return null;
    }
}
